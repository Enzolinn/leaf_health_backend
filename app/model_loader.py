# backend/app/model_loader.py
import torch
import os
import time
from typing import Any

class ModelManager:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._mtime = None
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self._mtime = os.path.getmtime(self.model_path)
        # substitua por código de loading do seu framework
        self._model = torch.load(self.model_path, map_location="cpu")
        # se usando PyTorch Lightning ou state_dict, ajuste aqui
        print(f"[ModelManager] loaded model from {self.model_path} (mtime={self._mtime})")

    def get_model(self) -> Any:
        # checa se o arquivo mudou (treino sobrescreveu)
        try:
            mtime = os.path.getmtime(self.model_path)
            if mtime != self._mtime:
                print("[ModelManager] model file changed, reloading...")
                self._load()
        except FileNotFoundError:
            pass
        return self._model
