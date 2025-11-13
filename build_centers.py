"""
Como usar:
  python build_centers.py --output centers_lab_global.npy
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from segmentation_kmeans import detect_lesions_kmeans


DATASET_ROOT = Path("dataset/datasets/vipoooool/new-plant-diseases-dataset/versions/2/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)")


def segment_leaf_hsv(image_rgb):
    """Retorna máscara binária da folha"""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def collect_lab_centers(image_path, k_local=3):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    leaf_mask = segment_leaf_hsv(img)
    if leaf_mask.sum() < 500:  # se a folha é muito pequena ou não detectada
        return None

    # converte para LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_leaf = lab[leaf_mask > 0].reshape(-1, 3)

    # MiniBatchKMeans por folha
    km = MiniBatchKMeans(n_clusters=k_local, random_state=0, n_init=10)
    km.fit(lab_leaf)
    centers = km.cluster_centers_
    return centers  # (k_local, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default="centers_lab_global.npy", help="Arquivo de saída")
    ap.add_argument("--k_local", type=int, default=3, help="Clusters por folha")
    ap.add_argument("--k_global", type=int, default=8, help="Clusters globais de cor")
    args = ap.parse_args()

    data_dir = DATASET_ROOT / "train"
    image_paths = list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png")) + list(data_dir.rglob("*.jpeg"))
    print(f"[+] Encontradas {len(image_paths)} imagens em {data_dir}")
    if len(image_paths) == 0:
        print("Nenhuma imagem encontrada.")
        return

    all_centers = []
    for p in tqdm(image_paths, desc="Coletando centros locais"):
        centers = collect_lab_centers(p, k_local=args.k_local)
        if centers is not None:
            all_centers.append(centers)

    if len(all_centers) == 0:
        print("Nenhum centro coletado.")
        return

    all_centers = np.concatenate(all_centers, axis=0)
    print(f"[+] Total de {len(all_centers)} centros coletados.")

    # Clustering global
    print(f"[+] Executando KMeans global (k={args.k_global})...")
    global_km = MiniBatchKMeans(n_clusters=args.k_global, random_state=0, n_init=10)
    global_km.fit(all_centers)
    centers_lab_global = global_km.cluster_centers_
    np.save(args.output, centers_lab_global)
    print(f"[✓] Centros salvos em {args.output}")

if __name__ == "__main__":
    main()
