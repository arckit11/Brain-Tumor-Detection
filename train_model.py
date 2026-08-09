"""
Train a ResNet50 model for brain tumor classification.

Produces models/bt_resnet50_model.pt with the exact architecture
expected by app.py (4-class: None, Meningioma, Glioma, Pituitary).

Usage:
    python train_model.py
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models


# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = "./brain_tumor_dataset/Brain Tumor"
MODEL_SAVE_PATH = "./models/bt_resnet50_model.pt"
NUM_CLASSES = 4
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
IMAGE_SIZE = 224          # Standard ResNet input size (model uses AdaptiveAvgPool)
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class mapping: folder name → index  (must match app.py LABELS)
# app.py: LABELS = ['None', 'Meningioma', 'Glioma', 'Pitutary']
#   index 0 = 'None'       → dataset folder 'notumor'
#   index 1 = 'Meningioma' → dataset folder 'meningioma'
#   index 2 = 'Glioma'     → dataset folder 'glioma'
#   index 3 = 'Pitutary'   → dataset folder 'pituitary'
CLASS_TO_IDX = {
    "notumor": 0,
    "meningioma": 1,
    "glioma": 2,
    "pituitary": 3,
}

TRAIN_RATIO = 0.85  # 85% train, 15% validation


# ─── Data Transforms ─────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Build Model (EXACT architecture from app.py) ────────────────────────────
def build_model():
    """
    Replicate the exact architecture defined in app.py lines 27-40:
      resnet50 backbone  →  custom fc head
        Linear(2048, 2048) → SELU → Dropout(0.4)
        Linear(2048, 2048) → SELU → Dropout(0.4)
        Linear(2048, 4)    → LogSigmoid
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    n_inputs = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 2048),
        nn.SELU(),
        nn.Dropout(p=0.4),
        nn.Linear(2048, 2048),
        nn.SELU(),
        nn.Dropout(p=0.4),
        nn.Linear(2048, NUM_CLASSES),
        nn.LogSigmoid(),
    )

    # Unfreeze all parameters (fine-tune entire network)
    for param in model.parameters():
        param.requires_grad = True

    return model


# ─── Dataset wrapper to apply different transforms ───────────────────────────
class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset to apply a custom transform."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scaler = GradScaler("cuda")  # Mixed precision for memory efficiency

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    with autocast("cuda"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Progress indicator every 20 batches
                if phase == "train" and (batch_idx + 1) % 20 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(dataloaders[phase])} | "
                          f"Loss: {loss.item():.4f}")

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"  {phase.capitalize():5s} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            # Save best model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"  ✓ New best val accuracy: {best_acc:.4f}")

    elapsed = time.time() - since
    print(f"\n{'=' * 40}")
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best val accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Locate dataset ──
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Dataset directory not found at {DATA_DIR}")
        print("Make sure the dataset is extracted under brain_tumor_dataset/Brain Tumor/")
        return

    # ── Load full dataset (without transforms — we apply them per-split) ──
    print("\nLoading dataset...")
    # Use a basic transform just for loading (resize + to PIL)
    base_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ])
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=base_transform)

    # Re-map class indices to match app.py LABELS order
    # ImageFolder assigns indices alphabetically: glioma=0, meningioma=1, notumor=2, pituitary=3
    # But app.py expects: notumor=0, meningioma=1, glioma=2, pituitary=3
    folder_to_alpha_idx = full_dataset.class_to_idx
    print(f"ImageFolder class_to_idx: {folder_to_alpha_idx}")

    remap = {}
    for folder_name, alpha_idx in folder_to_alpha_idx.items():
        remap[alpha_idx] = CLASS_TO_IDX[folder_name]
    print(f"Remapping: {remap}")

    # Apply remapping
    full_dataset.samples = [(path, remap[label]) for path, label in full_dataset.samples]
    full_dataset.targets = [remap[t] for t in full_dataset.targets]

    # ── Train/Val split ──
    total = len(full_dataset)
    train_size = int(TRAIN_RATIO * total)
    val_size = total - train_size
    print(f"Total: {total} | Train: {train_size} | Val: {val_size}")

    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Wrap subsets with appropriate transforms
    train_dataset = TransformSubset(train_subset, transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]))

    val_dataset = TransformSubset(val_subset, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]))

    # ── Data loaders ──
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True),
    }
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    # ── Build model ──
    print("\nBuilding ResNet50 model...")
    model = build_model().to(DEVICE)

    # ── Loss, optimizer, scheduler ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # ── Train ──
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    model = train_model(model, dataloaders, dataset_sizes,
                        criterion, optimizer, scheduler, NUM_EPOCHS)

    # ── Save ──
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    file_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    print(f"\n✓ Model saved to {MODEL_SAVE_PATH} ({file_size_mb:.1f} MB)")
    print("You can now run the Flask app with: python app.py")


if __name__ == "__main__":
    main()
