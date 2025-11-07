#!/usr/bin/env python3
"""
app.py - FastAPI inference server for plant disease model
Retorna:
{
  "plant": "Tomato",
  "disease": "Target_Spot",
  "scores": {...},
  "gradcam_png_b64": "...",
  "mask_png_b64": "..."
}
"""
import io, base64
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ===============================
# CONFIGURAÇÕES
# ===============================
MODEL_PATH = Path("work/best_model.pth")
USE_CUDA = True
IMG_SIZE = 224
MASK_THRESHOLD = 0.5   # pixel > 0.5 = área destacada
# ===============================

app = FastAPI(title="Plant Disease Detector")
device = "cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu"

# ------------------------------
# CARREGAR MODELO
# ------------------------------
def load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado em {path}")
    ck = torch.load(str(path), map_location=device)
    return ck

def build_model(ck):
    class_to_idx = ck["class_to_idx"]
    num_classes = len(class_to_idx)
    try:
        model = models.efficientnet_b0(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features, num_classes)
        )
    except Exception:
        model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(ck["model_state"])
    return model, class_to_idx

ckpt = load_checkpoint(MODEL_PATH)
model, class_to_idx = build_model(ckpt)
idx_to_class = {v: k for k, v in class_to_idx.items()}
model = model.to(device)
model.eval()

print(f"[+] Modelo carregado com {len(class_to_idx)} classes.")

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
    img = img.convert("RGB")
    return transform(img).unsqueeze(0)

# ------------------------------
# GRAD-CAM
# ------------------------------
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.target_layer = self._find_last_conv_layer()
        self._register_hooks()

    def _find_last_conv_layer(self):
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx):
        self.model.zero_grad()
        out = self.model(input_tensor)
        score = out[:, class_idx]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

gradcam = GradCAM(model)

# ------------------------------
# ROTA /predict
# ------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao abrir imagem: {e}")

    input_tensor = preprocess(img).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        label = idx_to_class[pred_idx]
        scores = {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}

    # separa planta e doença
    if "___" in label:
        plant, disease = label.split("___", 1)
    else:
        plant, disease = "unknown", label

    # gera grad-cam e máscara binária
    cam = gradcam(input_tensor, pred_idx)
    orig = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (0.45 * heatmap_color + 0.55 * orig).astype(np.uint8)

    # Máscara binária
    mask_bin = (cam > MASK_THRESHOLD).astype(np.uint8) * 255
    mask_rgb = np.stack([mask_bin]*3, axis=-1)

    # converte em base64
    def to_b64(arr):
        img_pil = Image.fromarray(arr)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    gradcam_b64 = to_b64(overlay)
    mask_b64 = to_b64(mask_rgb)

    return JSONResponse({
        "plant": plant,
        "disease": disease,
        "scores": scores,
        "gradcam_png_b64": gradcam_b64,
        "mask_png_b64": mask_b64
    })
