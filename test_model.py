# backend/test_model.py
import argparse
import json
import os
from pathlib import Path
from PIL import Image
import base64
import io

from app.model_loader import ModelManager
from app.infer import run_pipeline

def decode_base64_mask(mask_base64, output_path):
    """
    Converte o base64 (data URI) recebido em PNG e salva no disco.
    """
    if mask_base64.startswith("data:image"):
        header, encoded = mask_base64.split(",", 1)
    else:
        encoded = mask_base64
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes))
    img.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Teste local do modelo de detecção de doenças em folhas")
    parser.add_argument("--image", "-i", required=True, help="Caminho da imagem a ser testada")
    parser.add_argument("--output_dir", "-o", default="outputs", help="Diretório para salvar resultados")
    args = parser.parse_args()

    img_path = Path(args.image).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(__file__).resolve().parent / "models" / "latest_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

    print(f"🔹 Carregando modelo de: {model_path}")
    model_mgr = ModelManager(str(model_path))
    model = model_mgr.get_model()

    print(f"🔹 Rodando inferência na imagem: {img_path}")
    result = run_pipeline(str(img_path), model)

    # Mostra resultado em formato JSON legível
    print("\n===== RESULTADO JSON =====")
    print(json.dumps(result, indent=4))

    # Salva as máscaras (se houver)
    for pred in result.get("predictions", []):
        if "mask_base64" in pred:
            out_mask = output_dir / f"{Path(img_path).stem}_mask_{pred['id']}.png"
            decode_base64_mask(pred["mask_base64"], out_mask)
            print(f"🖼️  Máscara salva em: {out_mask}")

    print("\n✅ Teste concluído com sucesso!")

if __name__ == "__main__":
    main()
