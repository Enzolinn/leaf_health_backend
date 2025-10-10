# backend/app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import uuid
import os
from model_loader import ModelManager
from infer import run_pipeline

app = FastAPI(title="Plant Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção limite aos domínios do app
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.abspath(os.path.join(MODEL_DIR, "latest_model.pt"))

model_mgr = ModelManager(model_path=MODEL_PATH)  # carrega modelo inicial

UPLOAD_TMP = "/tmp/plant_images"
os.makedirs(UPLOAD_TMP, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...), crop_type: str = None):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image.")
    # salva temporariamente
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    tmp_path = os.path.join(UPLOAD_TMP, filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # assegura modelo carregado (se foi trocado no disco ele recarrega)
    model = model_mgr.get_model()

    # pipeline de inferencia: implementar em infer.run_pipeline
    # deve retornar a estrutura de dados já no formato COCO-like
    result = run_pipeline(img_path=tmp_path, model=model, crop_type=crop_type)

    # cleanup (opcional)
    try:
        os.remove(tmp_path)
    except:
        pass

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
