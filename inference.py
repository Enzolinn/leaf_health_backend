#!/usr/bin/env python3
"""
app.py - FastAPI inference server (classificação + detecção por inversão de cor)
Detecta doença invertendo as áreas saudáveis (baseado em cor dominante da folha).
Salva cópias locais para debug:
 - original.png
 - leaf_mask.png
 - healthy_mask.png
 - disease_mask.png
 - overlay.png
"""

import io, base64, os, datetime
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import segmentation_kmeans

# ===============================
# CONFIGURAÇÕES
# ===============================
MODEL_PATH = Path("work/best_model_a100.pth")
USE_CUDA = True
IMG_SIZE = 224

HSV_LOWER = np.array([20, 40, 30], dtype=np.uint8)
HSV_UPPER = np.array([100, 255, 255], dtype=np.uint8)

HEALTHY_PERCENTILE = 60  # quanto menor, mais sensível à doença
GAUSSIAN_BLUR_K = 5
MORPH_KERNEL = 5
MIN_COMPONENT_AREA = 100

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ===============================
app = FastAPI(title="Plant Disease Detector - Inverted Color Analysis")
device = "cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu"

# ------------------------------
# MODELO
# ------------------------------
def load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado em {path}")
    return torch.load(str(path), map_location=device)

def build_model(ck):
    class_to_idx = ck["class_to_idx"]
    num_classes = len(class_to_idx)
    model = models.efficientnet_b3(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(ck["model_state"])
    return model, class_to_idx

ckpt = load_checkpoint(MODEL_PATH)
model, class_to_idx = build_model(ckpt)
idx_to_class = {v: k for k, v in class_to_idx.items()}
model = model.to(device).eval()

print(f"[+] Modelo carregado com {len(class_to_idx)} classes. Device: {device}")

# ------------------------------
# TRANSFORMAÇÃO
# ------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

def preprocess(img: Image.Image):
    return transform(img.convert("RGB")).unsqueeze(0)

# ------------------------------
# UTILIDADES
# ------------------------------
def save_image_local(path, np_img):
    cv2.imwrite(str(path), cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))

def keep_largest_component_uint8(mask_uint8, min_area=50):
    nb, out, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if nb <= 1:
        return mask_uint8
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + np.argmax(areas)
    return ((out == largest).astype(np.uint8) * 255)

def bbox_from_mask_uint8(mask_uint8):
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0,0,0,0]
    x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return [int(x), int(y), int(w), int(h)]

def pil_to_b64_png(arr):
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ------------------------------
# SEGMENTAÇÃO DE FOLHA
# ------------------------------
def segment_leaf_hsv(orig_rgb_np):
    hsv = cv2.cvtColor(orig_rgb_np, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return keep_largest_component_uint8(mask, min_area=500)



# ------------------------------
# ENDPOINT /predict
# ------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = OUTPUT_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # abre imagem
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao abrir imagem: {e}")

    orig_np = np.array(img)
    save_image_local(out_dir / "original.png", orig_np)

    # inferência
    input_tensor = preprocess(img).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        label = idx_to_class[pred_idx]
        scores = {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}

    plant, disease = label.split("___", 1) if "___" in label else ("unknown", label)

    # segmentação da folha
    leaf_mask = segment_leaf_hsv(orig_np)
    save_image_local(out_dir / "leaf_mask.png", np.stack([leaf_mask]*3, axis=-1))

    # tenta carregar centers pré-treinados; se não existir, faz fit local
    pretrained_centers = None
    try:
        pretrained_centers = np.load("centers_lab_global.npy")
        use_minibatch = False
        k_for_call = len(pretrained_centers)
    except Exception:
        # arquivo não encontrado ou erro ao carregar -> teremos fit por imagem (k padrão)
        pretrained_centers = None
        use_minibatch = True
        k_for_call = 3  # valor padrão (pode ajustar)
        print("[!] centers_lab_global.npy não encontrado — usando KMeans por imagem (fit local).")

    # chama a função do módulo; descartamos os retornos que não usamos com '_'
    lesion_mask, cluster_map, bbox, area_pct, cluster_vis, centers_lab = segmentation_kmeans.detect_lesions_kmeans(
        orig_np,
        leaf_mask,
        k=k_for_call,
        use_minibatch=use_minibatch,
        pretrained_centers=pretrained_centers
    )

    # compute damage relative to leaf
    damage_on_leaf = int(((lesion_mask>0) & (leaf_mask>0)).sum())
    leaf_area = int((leaf_mask>0).sum())
    damage_pct_of_leaf = 100.0 * damage_on_leaf / leaf_area if leaf_area>0 else 0.0

    # save debug images:
    save_image_local(out_dir/"cluster_vis.png", cluster_vis)
    save_image_local(out_dir/"lesion_mask.png", np.stack([lesion_mask]*3, axis=-1))

    # base64
    leaf_b64 = pil_to_b64_png(np.stack([leaf_mask]*3, axis=-1))
    cluster_vis_b64 = pil_to_b64_png(cluster_vis)

    # resposta
    return JSONResponse({
        "plant": plant,
        "disease": disease,
        "scores": scores,
        "leaf_mask_b64": leaf_b64,
        "leaf_cluster_vis_b64": cluster_vis_b64,
        "damage_amount": damage_pct_of_leaf
    })

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)
