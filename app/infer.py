# backend/app/infer.py
import json
import numpy as np
from typing import Dict
from PIL import Image
import base64
import io
import os

# Implementar funções utilitárias reais: yolo_detect, deeplab_segment, classify, unet_segment
# Abaixo são placeholders/contratos.

def yolo_detect(img: Image.Image):
    # retorna lista de bboxes [ (x,y,w,h), ... ]
    return [(0,0,img.width,img.height)]

def deeplab_segment(crop_img: Image.Image):
    # retorna binary mask (numpy array HxW boolean)
    return np.ones((crop_img.height, crop_img.width), dtype=bool)

def classify(crop_img: Image.Image, model):
    # retorna (label_id, label_name, score)
    return (1, "Tomato Bacterial Spot", 0.93)

def unet_segment(crop_img: Image.Image, model):
    # retorna binary mask numpy HxW
    return np.zeros((crop_img.height, crop_img.width), dtype=bool)

def mask_to_base64_png(mask_np):
    # converte máscara binária para PNG RGBA (ou retorne RLE)
    h, w = mask_np.shape
    img = Image.fromarray((mask_np.astype('uint8')*255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def run_pipeline(img_path: str, model, crop_type: str = None) -> Dict:
    img = Image.open(img_path).convert("RGB")
    detections = []
    for bbox in yolo_detect(img):
        x,y,w,h = bbox
        crop = img.crop((x, y, x+w, y+h)).resize((256,256))
        leaf_mask = deeplab_segment(crop)
        label_id, label_name, score = classify(crop, model)
        lesion_mask = unet_segment(crop, model)
        affected = lesion_mask.sum()
        total = leaf_mask.sum() if leaf_mask.sum()>0 else crop.width*crop.height
        severity = float(affected) / float(total) * 100.0
        predictions = {
            "id": len(detections)+1,
            "category_id": label_id,
            "category_name": label_name,
            "score": float(score),
            "bbox": [int(x), int(y), int(w), int(h)],
            "mask_base64": mask_to_base64_png(lesion_mask),
            "severity_pct": severity,
            "remedy": "Apply copper-based bactericide"
        }
        detections.append(predictions)

    return {
        "image_id": os.path.basename(img_path),
        "width": img.width,
        "height": img.height,
        "predictions": detections,
        "model_version": getattr(model, "version", "unknown")
    }
