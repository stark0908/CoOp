import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import clip
from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Hyperparameters (per paper) ──────────────────────────────────
NUM_SHOTS = 16
EPOCHS = 200        # paper: 200 for 16-shot
LR = 0.002          # paper: SGD with lr=0.002
WARMUP_LR = 1e-5    # paper: warmup lr for first epoch
N_CTX = 4           # context length

# ── Data ─────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class EuroSATFewShot(Dataset):
    """Few-shot subset: only `shots` samples per base class."""
    def __init__(self, root, base_classes, shots=16):
        self.dataset = EuroSAT(root=root, transform=transform)

        self.indices = []
        class_counts = {c: 0 for c in base_classes}

        for i, (_, label) in enumerate(self.dataset):
            if label in base_classes and class_counts[label] < shots:
                self.indices.append(i)
                class_counts[label] += 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


# Load full dataset & split classes
dataset = EuroSAT(root="./data", download=True)

all_classes = list(set([label for _, label in dataset]))
random.shuffle(all_classes)

split = len(all_classes) // 2
base_classes = all_classes[:split]
new_classes = all_classes[split:]

# Build class name mapping from dataset folder names
EUROSAT_CLASSES = {i: name.lower().replace("_", " ") for i, name in enumerate(dataset.classes)}

print(f"Base classes ({len(base_classes)}): {[EUROSAT_CLASSES[c] for c in base_classes]}")
print(f"New  classes ({len(new_classes)}): {[EUROSAT_CLASSES[c] for c in new_classes]}")

# Train loader: few-shot on base classes only
train_dataset = EuroSATFewShot("./data", base_classes, NUM_SHOTS)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Test loader: all samples (base + new)
test_dataset = EuroSAT(root="./data", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64)

# ── CLIP ─────────────────────────────────────────────────────────
clip_model, preprocess = clip.load("ViT-B/16", device=DEVICE)

for param in clip_model.parameters():
    param.requires_grad = False

# ── Model ────────────────────────────────────────────────────────

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=4):
        super().__init__()

        self.classnames = classnames
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx

        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Initialize context from "a photo of a"
        prompt = "a photo of a"
        tokenized = clip.tokenize(prompt).to(DEVICE)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(torch.float32)

        self.ctx = nn.Parameter(embedding[0, 1:1+n_ctx, :])

        # Tokenize class names
        self.tokenized_prompts = torch.cat([
            clip.tokenize(f"a photo of a {name}") for name in classnames
        ]).to(DEVICE)

        self.clip_model = clip_model

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        token_embeddings = self.clip_model.token_embedding(self.tokenized_prompts)

        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1+self.n_ctx:, :]

        prompts = torch.cat([prefix, ctx, suffix], dim=1)

        return prompts


class CoOpModel(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()

        self.prompt_learner = PromptLearner(classnames, clip_model, N_CTX)
        self.clip_model = clip_model

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner()
        text_features = self.clip_model.encode_text(prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T
        return logits


# ── Setup ────────────────────────────────────────────────────────
# Model trained on base classes only
classnames = [EUROSAT_CLASSES[c] for c in base_classes]
model = CoOpModel(classnames, clip_model).to(DEVICE)

optimizer = torch.optim.SGD(
    model.prompt_learner.parameters(),
    lr=LR
)

# Cosine annealing scheduler (paper: cosine decay over training)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

criterion = nn.CrossEntropyLoss()


# ── Zero-shot text features for new classes ──────────────────────
def get_zeroshot_text_features(classnames_list):
    """Get CLIP zero-shot text features for a list of class names."""
    prompts = [f"a photo of a {EUROSAT_CLASSES[c]}" for c in classnames_list]
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
    - Base classes: use learned prompts (model output indices 0..4)
    - New classes: use CLIP zero-shot text features
    All images classified against all 10 classes.

    Returns (base_acc, new_acc, harmonic_mean)
    """
    model.eval()

    # Get learned text features for base classes
    with torch.no_grad():
        prompts = model.prompt_learner()
        base_text_features = model.clip_model.encode_text(prompts).float()
        base_text_features = base_text_features / base_text_features.norm(dim=-1, keepdim=True)

    # Get zero-shot text features for new classes
    new_text_features = get_zeroshot_text_features(new_classes)

    # Combined: [base_0, ..., base_4, new_0, ..., new_4]
    all_text_features = torch.cat([base_text_features, new_text_features], dim=0)

    # Build label mapping: original_label → index in all_text_features
    all_classes_ordered = base_classes + new_classes
    label_to_idx = {c: i for i, c in enumerate(all_classes_ordered)}

    correct_base = 0
    total_base = 0
    correct_new = 0
    total_new = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"         [Eval] ", leave=False)
        for images, labels in pbar:
            images = images.to(DEVICE)

            # Get image features
            image_features = model.clip_model.encode_image(images).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Classify against all 10 classes
            logits = image_features @ all_text_features.T
            preds = logits.argmax(dim=1).cpu()

            for i in range(len(labels)):
                label = labels[i].item()

                if label in base_classes:
                    total_base += 1
                    if preds[i].item() == label_to_idx[label]:
                        correct_base += 1

                elif label in new_classes:
                    total_new += 1
                    if preds[i].item() == label_to_idx[label]:
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
    print(f"CoOp Training — {EPOCHS} epochs, {NUM_SHOTS}-shot")
    print(f"Base: {base_classes} | New: {new_classes}")
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
            # Restore lr for scheduler to take over
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
