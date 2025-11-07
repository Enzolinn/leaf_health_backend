#!/usr/bin/env python3
import os
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torch.optim as optim

# ==============================
# CONFIGURAÇÕES EDITÁVEIS
# ==============================

# Caminho até o diretório "New Plant Diseases Dataset(Augmented)"
# Ajuste conforme o seu diretório exato
DATASET_PATH = Path("dataset/datasets/vipoooool/new-plant-diseases-dataset/versions/2/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)")

WORK_DIR = Path("work")            # onde salvar modelo e logs
EPOCHS = 20                        # número de épocas de treino
BATCH_SIZE = 16                    # tamanho do batch
LEARNING_RATE = 1e-3               # taxa de aprendizado
USE_CUDA = True                    # usar GPU se disponível
BACKBONE = "efficientnet_b0"       # ou "resnet50"
GRAD_CLIP = 1.0                    # limite de gradiente
WEIGHT_DECAY = 1e-4                # regularização
MODEL_OUT = WORK_DIR / "best_model.pth"  # arquivo final
# ==============================

WORK_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu"
print(f"[+] Dispositivo: {device}")

# ------------------------------
# PREPARAR DATASET
# ------------------------------
train_dir = DATASET_PATH / "train"
val_dir = DATASET_PATH / "valid"

if not train_dir.exists() or not val_dir.exists():
    raise FileNotFoundError(f"Diretórios 'train' e 'valid' não encontrados em {DATASET_PATH}")

print(f"[+] Usando dataset em: {DATASET_PATH}")

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

num_classes = len(train_ds.classes)
print(f"[+] Classes detectadas: {num_classes}")

# ------------------------------
# DEFINIR MODELO
# ------------------------------
if BACKBONE == "efficientnet_b0":
    model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
else:
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

# ------------------------------
# FUNÇÃO DE AVALIAÇÃO
# ------------------------------
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

# ------------------------------
# LOOP DE TREINAMENTO
# ------------------------------
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Época {epoch}/{EPOCHS}")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        if GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=running_loss / total, acc=correct / total)

    train_acc = correct / total
    val_acc = evaluate(val_loader)
    scheduler.step(1 - val_acc)

    print(f"Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state': model.state_dict(),
            'class_to_idx': train_ds.class_to_idx,
            'val_acc': val_acc,
            'epoch': epoch
        }, MODEL_OUT)
        print(f"[+] Novo melhor modelo salvo: {MODEL_OUT}")

print(f"[✓] Treinamento concluído! Melhor acurácia de validação: {best_val_acc:.4f}")
