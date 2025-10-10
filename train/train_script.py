# backend/train/train_script.py (esqueleto)
import torch
import os
from pathlib import Path

# supondo que você tenha um objeto `model` já treinado
def train():
    # treine seu modelo: loaders, optimizer, epochs...
    # ...
    model = ...  # seu modelo pronto
    model.version = "v2025-10-09-001"
    return model

if __name__ == "__main__":
    model = train()
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "latest_model.pt"
    # se o seu modelo for state_dict:
    # torch.save(model.state_dict(), save_path)
    # ou salve o objeto inteiro (menos recomendado p/ compatibilidade)
    torch.save(model, save_path)
    print(f"Model saved to {save_path}")
