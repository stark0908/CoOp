import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import clip
from collections import OrderedDict
from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
try:
    import wandb
except Exception:
    wandb = None

DEVICE = os.environ.get("CUDA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Path to Eurosat data (override with env var `EUROSAT_ROOT`).
# Default to torchvision-style local folder and allow automatic download.
DATA_ROOT = os.environ.get("EUROSAT_ROOT", "./data")

# ── Hyperparameters (per paper) ──────────────────────────────────
NUM_SHOTS = 16
EPOCHS = 10         # paper: 10 epochs for CoCoOp
LR = 0.002          # paper: SGD with lr=0.002
WARMUP_LR = 1e-5    # paper: warmup lr for first epoch
N_CTX = 4           # context length

# ── Data (paper: RandomResizedCrop + RandomFlip + CLIP normalize) ─
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

test_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


class EuroSATFewShot(Dataset):
    """Few-shot subset: only `shots` samples per base class."""
    def __init__(self, root, base_classes, shots=16, transform=None):
        self.dataset = EuroSAT(root=root, transform=None)
        self.transform = transform

        self.indices = []
        class_counts = {c: 0 for c in base_classes}

        # ⚡ FAST: no image loading
        for i, label in enumerate(self.dataset.targets):
            if label in base_classes and class_counts[label] < shots:
                self.indices.append(i)
                class_counts[label] += 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


# Load full dataset & split classes (use torchvision's downloader)
dataset = EuroSAT(root=DATA_ROOT, download=True)

all_classes = list(set([label for _, label in dataset]))
random.shuffle(all_classes)

split = len(all_classes) // 2
base_classes = all_classes[:split]
new_classes = all_classes[split:]

# Build class name mapping from dataset folder names
EUROSAT_CLASSES = {i: name.lower().replace("_", " ") for i, name in enumerate(dataset.classes)}

print(f"Base classes ({len(base_classes)}): {[EUROSAT_CLASSES[c] for c in base_classes]}")
print(f"New  classes ({len(new_classes)}): {[EUROSAT_CLASSES[c] for c in new_classes]}")

# Use ordered list of all classes (base + new) for model / label mapping
all_classes = base_classes + new_classes

# Global label maps (computed once)
# label_map_all: dataset class_id -> position in model outputs (all_classes ordering)
label_map_all = {c: i for i, c in enumerate(all_classes)}
# base_label_rel: dataset class_id (only for base classes) -> 0..(n_base-1)
base_label_rel = {c: i for i, c in enumerate(base_classes)}

# Train loader: few-shot on base classes only (with augmentation)
# Batch size = 1 for deterministic per-instance prompts
num_workers = min(8, (os.cpu_count() or 1))
pin_memory = True
train_dataset = EuroSATFewShot(DATA_ROOT, base_classes, NUM_SHOTS, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)

# Test loader: all samples (no augmentation)
test_dataset = EuroSAT(root=DATA_ROOT, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64,
                         num_workers=num_workers, pin_memory=pin_memory)

print("Data loaders created.", flush=True)

# ── CLIP ─────────────────────────────────────────────────────────
print("Loading CLIP model...", flush=True)
clip_model, _ = clip.load("ViT-B/16", device=DEVICE)
print("CLIP model loaded.", flush=True)

for param in clip_model.parameters():
    param.requires_grad = False

# ── Custom TextEncoder (matches original repo) ──────────────────

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Take features from the EOT embedding
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


# ── Model ────────────────────────────────────────────────────────

class PromptLearnerCoCoOp(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=4):
        super().__init__()

        self.n_cls = len(classnames)
        self.n_ctx = n_ctx

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim

        # Initialize context from "a photo of a"
        prompt = "a photo of a"
        tokenized = clip.tokenize(prompt).to(DEVICE)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(dtype)

        # Keep learnable context parameters in FP32 to avoid FP16-grad issues with GradScaler
        self.ctx = nn.Parameter(embedding[0, 1:1+n_ctx, :].to(torch.float32))

        # Meta-Net: vis_dim → vis_dim//16 → ctx_dim (matches original repo)
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        # Keep meta-net in FP32 (GradScaler cannot unscale FP16 grads)
        self.meta_net = self.meta_net.to(torch.float32)
        self.meta_net_dtype = torch.float32
        self.dtype = dtype

        # Tokenize class prompts (with period, matching original)
        classnames_clean = [name.replace("_", " ") for name in classnames]
        prompts_text = [f"a photo of a {name}." for name in classnames_clean]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_text]).to(DEVICE)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # Store prefix (SOS) and suffix (CLASS + . + EOS) as buffers
        self.register_buffer("token_prefix", embedding[:, :1, :])          # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, ., EOS

        self.tokenized_prompts = tokenized_prompts

    def construct_prompts(self, ctx, prefix, suffix):
        # Ensure prefix/suffix match ctx dtype (ctx is FP32)
        # kept for compatibility; not used by the vectorized forward
        prefix = prefix.to(ctx.dtype)
        suffix = suffix.to(ctx.dtype)
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, image_features):
        """
        image_features: (B, D)
        Returns prompts: (B, C, L, D)
        """
        prefix = self.token_prefix                          # (C, 1, D)
        suffix = self.token_suffix                          # (C, S, D)
        ctx = self.ctx                                      # (n_ctx, D)
        # meta_net parameters are FP32; ensure inputs are cast to FP32 to avoid FP16-grad issues
        bias = self.meta_net(image_features.to(self.meta_net_dtype))    # (B, ctx_dim)
        bias = bias.unsqueeze(1)                            # (B, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)                              # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias                            # (B, n_ctx, ctx_dim)

        # Vectorized expansion across classes:
        B = ctx_shifted.shape[0]
        prefix_exp = prefix.unsqueeze(0).expand(B, -1, -1, -1).to(ctx_shifted.dtype)   # (B, C, 1, D)
        ctx_exp = ctx_shifted.unsqueeze(1).expand(-1, self.n_cls, -1, -1)             # (B, C, n_ctx, D)
        suffix_exp = suffix.unsqueeze(0).expand(B, -1, -1, -1).to(ctx_shifted.dtype)  # (B, C, S, D)

        prompts = torch.cat([prefix_exp, ctx_exp, suffix_exp], dim=2)  # (B, C, L, D)
        return prompts.to(self.dtype)


class CoCoOpModel(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()

        self.prompt_learner = PromptLearnerCoCoOp(classnames, clip_model, N_CTX)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, images):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(images.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)   # (B, D)

        # Instance-conditional prompts (B, C, L, D)
        prompts = self.prompt_learner(image_features)

        B, C, L, D = prompts.shape

        # Flatten prompts and repeat tokenized_prompts for batch to encode in one call
        prompts_flat = prompts.view(B * C, L, D)
        tokens_expanded = tokenized_prompts.unsqueeze(0).expand(B, -1, -1).reshape(B * C, L).to(prompts_flat.device)
        text_features_flat = self.text_encoder(prompts_flat, tokens_expanded)  # (B*C, D)
        text_features = text_features_flat.view(B, C, -1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Batch matrix multiply: (B,1,D) @ (B,D,C) -> (B,1,C) -> squeeze -> (B,C)
        logits = logit_scale * (image_features.unsqueeze(1) @ text_features.transpose(2, 1)).squeeze(1)
        return logits


# ── Setup ────────────────────────────────────────────────────────
print("Setting up CoCoOp model (all classes)")
classnames = [EUROSAT_CLASSES[c] for c in all_classes]

model = CoCoOpModel(classnames, clip_model).to(DEVICE)

optimizer = torch.optim.SGD(
    model.prompt_learner.parameters(),
    lr=LR
)

# Cosine annealing scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

criterion = nn.CrossEntropyLoss()

# AMP (automatic mixed precision) support when running on CUDA
USE_AMP = torch.cuda.is_available() and ("cuda" in DEVICE)
if USE_AMP:
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()
else:
    scaler = None

# Initialize wandb (optional)
if wandb is not None:
    wandb.init(project=os.environ.get("WANDB_PROJECT", "cocoop"),
               config={"epochs": EPOCHS, "lr": LR, "warmup_lr": WARMUP_LR,
                       "n_ctx": N_CTX, "num_shots": NUM_SHOTS})
    try:
        wandb.watch(model, log="all", log_freq=100)
    except Exception:
        pass

# create fast mapping tensor
num_total_classes = max(base_label_rel.keys()) + 1
label_map_tensor = torch.full((num_total_classes,), -1, dtype=torch.long)

for k, v in base_label_rel.items():
    label_map_tensor[k] = v

label_map_tensor = label_map_tensor.to(DEVICE)

# create mapping for ALL classes (for eval)
num_total_classes_all = max(label_map_all.keys()) + 1
label_map_tensor_all = torch.full((num_total_classes_all,), -1, dtype=torch.long)

for k, v in label_map_all.items():
    label_map_tensor_all[k] = v

label_map_tensor_all = label_map_tensor_all.to(DEVICE)


# ── Training + Evaluation Loop ───────────────────────────────────
def train_one_epoch(epoch):
    """Train on base classes for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE)
        # labels are original dataset class ids (ints). Map to base-relative indices for loss.
        labels = labels.to(DEVICE)
        labels_rel = label_map_tensor[labels].to(DEVICE)

        # Forward / backward with autocast when using AMP
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                logits = model(images)                           # (B, n_all)
                logits_base = logits[:, :len(base_classes)]     # only base-class logits
                loss = criterion(logits_base, labels_rel)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            logits_base = logits[:, :len(base_classes)]
            loss = criterion(logits_base, labels_rel)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

    pbar.close()
    return total_loss / max(n_batches, 1)


base_classes_tensor = torch.tensor(base_classes, device=DEVICE)
new_classes_tensor  = torch.tensor(new_classes, device=DEVICE)


def evaluate():
    """
    Evaluate on ALL test data.
    CoCoOp generates instance-conditional prompts, so for base classes
    we use the learned prompt+meta-net via custom TextEncoder.
    For new classes we use zero-shot CLIP.

    All images classified against all 10 classes.
    Returns (base_acc, new_acc, harmonic_mean)
    """
    model.eval()

    correct_base = 0
    total_base = 0
    correct_new = 0
    total_new = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"         [Eval] ", leave=False)

        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # forward
            if USE_AMP:
                with torch.amp.autocast(device_type='cuda'):
                    logits_batch = model(images)
            else:
                logits_batch = model(images)

            preds = logits_batch.argmax(dim=1)

            # fast label mapping (NO python loop)
            label_indices = label_map_tensor_all[labels]

            # masks
            base_mask = torch.isin(labels, base_classes_tensor)
            new_mask  = torch.isin(labels, new_classes_tensor)

            # correctness
            correct = (preds == label_indices)

            # accumulate
            correct_base += (correct & base_mask).sum().item()
            total_base   += base_mask.sum().item()

            correct_new  += (correct & new_mask).sum().item()
            total_new    += new_mask.sum().item()

    base_acc = correct_base / max(total_base, 1)
    new_acc  = correct_new / max(total_new, 1)

    if base_acc + new_acc > 0:
        h = 2 * base_acc * new_acc / (base_acc + new_acc)
    else:
        h = 0.0

    return base_acc, new_acc, h


def train_and_evaluate():
    print(f"\n{'='*50}")
    print(f"CoCoOp Training — {EPOCHS} epochs, {NUM_SHOTS}-shot")
    print(f"Base: {[EUROSAT_CLASSES[c] for c in base_classes]}")
    print(f"New:  {[EUROSAT_CLASSES[c] for c in new_classes]}")
    print(f"{'='*50}\n")

    for epoch in range(EPOCHS):
        # Warmup: fix lr at 1e-5 for first epoch
        if epoch == 0:
            for pg in optimizer.param_groups:
                pg['lr'] = WARMUP_LR

        # Train
        avg_loss = train_one_epoch(epoch)

        # Step scheduler (after warmup epoch)
        if epoch == 0:
            for pg in optimizer.param_groups:
                pg['lr'] = LR
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Evaluate
        base_acc, new_acc, h = evaluate()

        # Log to stdout
        print(f"{'='*15} Epoch {epoch+1:3d}/{EPOCHS} {'='*15}")
        print(f"  Train Loss : {avg_loss:.4f}")
        print(f"  LR         : {current_lr:.6f}")
        print(f"  Base Acc   : {base_acc*100:.2f}%")
        print(f"  New Acc    : {new_acc*100:.2f}%")
        print(f"  H          : {h*100:.2f}%")
        print(f"{'='*42}")

        # Log to wandb if available
        if wandb is not None:
            wandb.log({
                "train/loss": avg_loss,
                "train/lr": current_lr,
                "eval/base_acc": base_acc,
                "eval/new_acc": new_acc,
                "eval/h": h,
                "epoch": epoch + 1,
                "timestamp": time.time()
            }, step=epoch + 1)

    # Finish wandb run
    if wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass


# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_evaluate()
