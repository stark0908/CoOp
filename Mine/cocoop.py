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

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

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

# Train loader: few-shot on base classes only (with augmentation)
train_dataset = EuroSATFewShot(DATA_ROOT, base_classes, NUM_SHOTS, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Test loader: all samples (no augmentation)
test_dataset = EuroSAT(root=DATA_ROOT, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64)

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
        prefix = prefix.to(ctx.dtype)
        suffix = suffix.to(ctx.dtype)
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, image_features):
        """
        image_features: (B, D)
        Returns prompts: (B, C, L, D)
        """
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                          # (n_ctx, ctx_dim)
        # meta_net parameters are FP32; ensure inputs are cast to FP32 to avoid FP16-grad issues
        bias = self.meta_net(image_features.to(self.meta_net_dtype))    # (batch, ctx_dim)
        bias = bias.unsqueeze(1)                # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)                  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias                # (batch, n_ctx, ctx_dim)

        # Build per-instance prompts for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)  # (B, C, L, D)

        return prompts


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
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Instance-conditional prompts
        prompts = self.prompt_learner(image_features)  # (B, C, L, D)

        # Per-image text encoding (necessary for CoCoOp)
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)  # (B, C)

        return logits


# ── Setup ────────────────────────────────────────────────────────
print("Setting up CoCoOp model (base classes only)")
classnames = [EUROSAT_CLASSES[c] for c in base_classes]

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


# ── Zero-shot text features for new classes ──────────────────────
def get_zeroshot_text_features(classnames_list):
    """Get CLIP zero-shot text features for a list of class names (with period)."""
    prompts = [f"a photo of a {EUROSAT_CLASSES[c]}." for c in classnames_list]
    tokens = clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


# ── Training + Evaluation Loop ───────────────────────────────────
def train_one_epoch(epoch):
    """Train on base classes for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE)
        # Map labels → base class indices (0..4)
        labels = torch.tensor([base_classes.index(l) for l in labels]).to(DEVICE)

        # Forward / backward with autocast when using AMP
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                logits = model(images)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

    pbar.close()
    return total_loss / max(n_batches, 1)


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

    # Zero-shot text features for new classes
    new_text_features = get_zeroshot_text_features(new_classes)

    # Build ordered class list: [base_0..base_4, new_0..new_4]
    all_classes_ordered = base_classes + new_classes
    label_to_idx = {c: i for i, c in enumerate(all_classes_ordered)}

    logit_scale = model.logit_scale.exp()
    tokenized_prompts = model.tokenized_prompts

    correct_base = 0
    total_base = 0
    correct_new = 0
    total_new = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"         [Eval] ", leave=False)
        for images, labels in pbar:
            images = images.to(DEVICE)
            B = images.shape[0]

            # Get image features (use autocast when AMP is enabled)
            if USE_AMP:
                with torch.amp.autocast(device_type='cuda'):
                    image_features = model.image_encoder(images.type(model.dtype)).float()
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    # Get instance-conditional prompts for base classes
                    prompts = model.prompt_learner(image_features)  # (B, n_base, L, D)
            else:
                image_features = model.image_encoder(images.type(model.dtype)).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                # Get instance-conditional prompts for base classes
                prompts = model.prompt_learner(image_features)  # (B, n_base, L, D)

            for i in range(B):
                # Base class text features (learned, instance-conditional)
                base_text_features = model.text_encoder(prompts[i], tokenized_prompts).float()
                base_text_features = base_text_features / base_text_features.norm(dim=-1, keepdim=True)

                # Combine with new class zero-shot features
                all_text_features = torch.cat([base_text_features, new_text_features], dim=0)

                # Classify against all classes (with logit_scale)
                logits = logit_scale * image_features[i] @ all_text_features.T
                pred = logits.argmax().item()

                label = labels[i].item()

                if label in base_classes:
                    total_base += 1
                    if pred == label_to_idx[label]:
                        correct_base += 1

                elif label in new_classes:
                    total_new += 1
                    if pred == label_to_idx[label]:
                        correct_new += 1

    base_acc = correct_base / max(total_base, 1)
    new_acc = correct_new / max(total_new, 1)

    # Harmonic mean
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

        # Log
        print(f"{'='*15} Epoch {epoch+1:3d}/{EPOCHS} {'='*15}")
        print(f"  Train Loss : {avg_loss:.4f}")
        print(f"  LR         : {current_lr:.6f}")
        print(f"  Base Acc   : {base_acc*100:.2f}%")
        print(f"  New Acc    : {new_acc*100:.2f}%")
        print(f"  H          : {h*100:.2f}%")
        print(f"{'='*42}")


# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_evaluate()
