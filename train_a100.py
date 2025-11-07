#!/usr/bin/env python3
"""
train_a100.py
Training script optimized for NVIDIA A100 (single GPU).
Features:
 - Automatic GPU detection (A100 -> use bfloat16 if supported)
 - Mixed precision (fp16 or bfloat16 via autocast)
 - OneCycleLR scheduler with linear scaling rule for LR
 - EfficientNet-B3/B4 (configurable) with transfer-learning head
 - Checkpointing (per-epoch) + best_model.pth saved
 - Optional freeze-backbone for initial epochs
"""

import os
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

# ------------------------
# USER-EDITABLE CONFIG
# ------------------------
# Dataset: point to the directory that contains 'train' and 'valid' subfolders
DATASET_ROOT = Path("dataset/datasets/vipoooool/new-plant-diseases-dataset/versions/2/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)")

WORK_DIR = Path("work_a100")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Model / training hyperparams
BACKBONE = "efficientnet_b3"    # options: efficientnet_b3, efficientnet_b4, resnet50, convnext_tiny
IMG_SIZE = 256                  # 224/256/320 might be good for A100
EPOCHS = 40
BATCH_SIZE = 128                # start 64/128 for A100 (adjust)
BASE_LR = 1e-3                  # base lr for reference batch (REF_BATCH below)
REF_BATCH = 32                  # reference batch for linear scaling
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
ACCUMULATION_STEPS = 1          # set >1 to simulate bigger batch: effective_batch = BATCH_SIZE * ACCUMULATION_STEPS
FREEZE_BACKBONE_EPOCHS = 0      # set >0 to freeze backbone for first N epochs
USE_AMP = True                  # enable mixed precision (bfloat16 on A100 if available; else fp16)
CLIP_GRAD_NORM = 1.0

NUM_WORKERS = 8
PIN_MEMORY = True
SEED = 42

# Checkpointing
CKPT_DIR = WORK_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = WORK_DIR / "best_model.pth"

# Misc
PRINT_FREQ = 1
SAVE_EPOCH_FREQ = 1
# ------------------------

# reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# device & detect GPU model
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "unknown"
else:
    gpu_name = "cpu"
print(f"[+] Device: {device} ({gpu_name})")

# bfloat16 availability check
USE_BFLOAT16 = False
if device == "cuda" and "A100" in gpu_name.upper():
    # A100 supports bfloat16 (hardware) and recent pytorch builds support autocast bfloat16
    # We'll use bfloat16 if USE_AMP is True and PyTorch supports it
    try:
        # Check if autocast accepts dtype=torch.bfloat16 in this build
        from torch.cuda.amp import autocast as _autocast
        # This is a heuristic; assume bfloat16 available on A100
        USE_BFLOAT16 = True
    except Exception:
        USE_BFLOAT16 = False

print(f"[+] USE_AMP={USE_AMP}, USE_BFLOAT16={USE_BFLOAT16}")

# ------------------------
# Data transforms & loaders
# ------------------------
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

train_dir = DATASET_ROOT / "train"
val_dir = DATASET_ROOT / "valid"
if not train_dir.exists() or not val_dir.exists():
    raise FileNotFoundError(f"train/valid not found under {DATASET_ROOT}")

train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

num_classes = len(train_ds.classes)
print(f"[+] Dataset: {len(train_ds)} train imgs, {len(val_ds)} val imgs, {num_classes} classes")

# ------------------------
# Model builder
# ------------------------
def build_model(backbone_name: str, num_classes: int):
    backbone_name = backbone_name.lower()
    if backbone_name.startswith("efficientnet_b3"):
        model = models.efficientnet_b3(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    elif backbone_name.startswith("efficientnet_b4"):
        model = models.efficientnet_b4(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    elif backbone_name.startswith("convnext"):
        model = models.convnext_tiny(pretrained=True)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    else:
        # fallback to resnet50
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model

model = build_model(BACKBONE, num_classes).to(device)

# optionally freeze backbone layers
if FREEZE_BACKBONE_EPOCHS > 0:
    print(f"[+] Freezing backbone params for first {FREEZE_BACKBONE_EPOCHS} epochs")
    for name, p in model.named_parameters():
        # crude: freeze all except classifier/fc/classifier layers
        if ("classifier" not in name) and ("fc" not in name):
            p.requires_grad = False

# ------------------------
# Optimizer & Scheduler (OneCycleLR)
# ------------------------
# linear scaling for lr
effective_batch = BATCH_SIZE * ACCUMULATION_STEPS
effective_lr = BASE_LR * (effective_batch / REF_BATCH)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=effective_lr, weight_decay=WEIGHT_DECAY)

# OneCycleLR: need steps_per_epoch
steps_per_epoch = max(1, len(train_loader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=effective_lr,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1,
    anneal_strategy='linear'
)

print(f"[+] Optimizer: AdamW, effective_batch={effective_batch}, effective_lr={effective_lr:.2e}")
print(f"[+] OneCycleLR steps_per_epoch={steps_per_epoch}, epochs={EPOCHS}")

criterion = nn.CrossEntropyLoss()

# amp scaler
scaler = GradScaler(enabled=(USE_AMP and not USE_BFLOAT16))

# training helpers
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss_sum += float(loss.item()) * labels.size(0)
    return correct / total if total > 0 else 0.0, loss_sum / total if total>0 else 0.0

best_val_acc = 0.0
global_step = 0

# choose autocast dtype
if USE_BFLOAT16:
    autocast_dtype = torch.bfloat16
else:
    autocast_dtype = torch.float16

print(f"[+] Training start. AUTCAST dtype: {autocast_dtype}")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    # optionally unfreeze after freeze period
    if FREEZE_BACKBONE_EPOCHS > 0 and epoch == FREEZE_BACKBONE_EPOCHS + 1:
        print("[*] Unfreezing backbone parameters.")
        for p in model.parameters():
            p.requires_grad = True
        # rebuild optimizer to include newly unfrozen params
        optimizer = optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=effective_lr,
            epochs=EPOCHS - epoch + 1,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='linear'
        )
        scaler = GradScaler(enabled=(USE_AMP and not USE_BFLOAT16))

    optimizer.zero_grad()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS}")
    for step, (imgs, labels) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixed precision: choose autocast dtype
        if USE_AMP:
            ctx = autocast(enabled=True, dtype=autocast_dtype)
        else:
            ctx = torch.cuda.amp.autocast(enabled=False)

        with ctx:
            outputs = model(imgs)
            loss = criterion(outputs, labels) / ACCUMULATION_STEPS

        # scale & backward (if using GradScaler)
        if USE_AMP and not USE_BFLOAT16:
            scaler.scale(loss).backward()
        else:
            # bfloat16 path or no amp
            loss.backward()

        # gradient accumulation step
        if (step + 1) % ACCUMULATION_STEPS == 0:
            if CLIP_GRAD_NORM and (CLIP_GRAD_NORM > 0):
                if USE_AMP and not USE_BFLOAT16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)

            if USE_AMP and not USE_BFLOAT16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            # update scheduler if OneCycleLR
            scheduler.step()
            global_step += 1

        # statistics
        preds = outputs.detach().argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)
        running_loss += float(loss.item()) * ACCUMULATION_STEPS * labels.size(0)

        if step % 50 == 0:
            pbar.set_postfix({"loss": running_loss / max(1, running_total), "acc": running_correct / max(1, running_total)})

    epoch_train_acc = running_correct / running_total if running_total > 0 else 0.0
    val_acc, val_loss = evaluate(model, val_loader, device)

    # checkpoint
    ckpt = {
        "model_state": model.state_dict(),
        "class_to_idx": train_ds.class_to_idx,
        "epoch": epoch,
        "val_acc": val_acc
    }
    ckpt_path = CKPT_DIR / f"ckpt_epoch_{epoch}.pth"
    torch.save(ckpt, ckpt_path)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(ckpt, BEST_PATH)
        print(f"[+] New best model saved to {BEST_PATH} (val_acc={val_acc:.4f})")

    t1 = time.time()
    print(f"Epoch {epoch} summary: train_acc={epoch_train_acc:.4f}, val_acc={val_acc:.4f}, val_loss={val_loss:.4f}, time={(t1-t0):.1f}s")

# final report
print(f"[✓] Training finished. Best val_acc = {best_val_acc:.4f}. Best model at {BEST_PATH}")

# Save training metadata
meta = {
    "best_val_acc": float(best_val_acc),
    "num_classes": num_classes,
    "backbone": BACKBONE,
    "img_size": IMG_SIZE,
    "epochs": EPOCHS
}
with open(WORK_DIR / "train_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
