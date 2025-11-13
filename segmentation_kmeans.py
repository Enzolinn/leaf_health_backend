# segmentation_kmeans.py
"""
Detect lesions inside a leaf mask using KMeans clustering in LAB color space.

Função principal:
 - detect_lesions_kmeans(orig_rgb_np, leaf_mask_uint8, ...)
   -> retorna (lesion_mask, cluster_vis)
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans

def _keep_largest_component_uint8(mask_uint8, min_area=50):
    nb, out, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if nb <= 1:
        return mask_uint8
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (out == largest).astype(np.uint8) * 255

def _bbox_from_mask_uint8(mask_uint8):
    contours, _ = cv2.findContours(mask_uint8.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0,0,0,0]
    x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return [int(x), int(y), int(w), int(h)]

def detect_lesions_kmeans(orig_rgb_np,
                          leaf_mask_uint8,
                          k=3,
                          use_minibatch=True,
                          minibatch_size=1000,
                          n_init=10,
                          random_state=0,
                          cluster_postproc_kernel=5,
                          keep_largest=True,
                          min_area=100,
                          pretrained_centers=None):
    """
    Detect lesions by clustering colors inside the leaf using KMeans (LAB space).

    Retorna:
      lesion_mask (HxW uint8 0/255), cluster_vis (HxWx3 uint8 RGB)
    """
    H, W = orig_rgb_np.shape[:2]
    leaf_idx = (leaf_mask_uint8 > 0)
    # caso não haja folha detectada, retornamos máscara vazia + visualização igual à imagem original
    if leaf_idx.sum() == 0:
        empty_mask = np.zeros((H,W), dtype=np.uint8)
        cluster_vis = orig_rgb_np.copy() if orig_rgb_np is not None else np.zeros((H,W,3), dtype=np.uint8)
        return empty_mask, cluster_vis

    # convert to LAB (uint8 -> float32)
    lab = cv2.cvtColor(orig_rgb_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_leaf = lab[leaf_idx].reshape(-1, 3)  # N x 3

    # optionally scale for stability
    mean = lab_leaf.mean(axis=0)
    std = lab_leaf.std(axis=0) + 1e-8
    lab_scaled = (lab_leaf - mean) / std

    # fit or use pretrained centers
    if pretrained_centers is None:
        if use_minibatch:
            mb = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=n_init)
            mb.fit(lab_scaled)
            labels = mb.predict(lab_scaled)
            centers_scaled = mb.cluster_centers_
        else:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
            labels = km.fit_predict(lab_scaled)
            centers_scaled = km.cluster_centers_
        centers_lab = centers_scaled * std.reshape(1,3) + mean.reshape(1,3)
    else:
        centers_lab = np.asarray(pretrained_centers, dtype=np.float32)
        centers_scaled = (centers_lab - mean.reshape(1,3)) / std.reshape(1,3)
        dif = lab_scaled[:,None,:] - centers_scaled[None,:,:]
        dists = np.linalg.norm(dif, axis=2)  # N x k
        labels = np.argmin(dists, axis=1)

    # build full cluster_map
    cluster_map = np.full((H,W), -1, dtype=np.int32)
    cluster_map[leaf_idx] = labels

    # determine healthy cluster (maior cluster)
    counts = np.bincount(labels, minlength=k if pretrained_centers is None else centers_lab.shape[0])
    healthy_cluster = int(np.argmax(counts))

    # lesion mask = leaf pixels with cluster != healthy_cluster
    lesion_mask = np.zeros((H,W), dtype=np.uint8)
    lesion_mask[np.logical_and(leaf_idx, cluster_map != healthy_cluster)] = 255

    # postprocess morphological
    if cluster_postproc_kernel and cluster_postproc_kernel > 0:
        kk = cluster_postproc_kernel if cluster_postproc_kernel % 2 == 1 else cluster_postproc_kernel + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk,kk))
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)

    # keep largest component opcional
    if keep_largest:
        lesion_mask = _keep_largest_component_uint8(lesion_mask, min_area=min_area)

    # cluster visualization: map each cluster to its center color (LAB->RGB)
    cluster_vis = np.zeros((H,W,3), dtype=np.uint8)
    if centers_lab is not None:
        # convert centers to uint8 LAB and then to RGB
        centers_lab_u8 = np.clip(centers_lab, 0, 255).astype(np.uint8).reshape(-1,1,3)
        centers_rgb = cv2.cvtColor(centers_lab_u8, cv2.COLOR_LAB2RGB).reshape(-1,3)
        # se cluster_map tem -1 fora da folha, só atribuímos cores dentro da folha
        for cid in range(len(centers_rgb)):
            cluster_vis[cluster_map == cid] = centers_rgb[cid]
    # fallback: manter a imagem original fora da folha
    cluster_vis[leaf_mask_uint8 == 0] = orig_rgb_np[leaf_mask_uint8 == 0]

    return lesion_mask, cluster_vis
