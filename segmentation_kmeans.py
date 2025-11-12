# segmentation_kmeans.py
"""
Detect lesions inside a leaf mask using KMeans clustering in LAB color space.

Functions:
 - detect_lesions_kmeans(...)
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

    Returns:
      lesion_mask (HxW uint8), cluster_map (HxW int), bbox [x,y,w,h], area_pct (float),
      cluster_vis (HxW3 uint8 RGB), centers_lab (k x 3 float)
    """
    H, W = orig_rgb_np.shape[:2]
    leaf_idx = (leaf_mask_uint8 > 0)
    if leaf_idx.sum() == 0:
        empty_mask = np.zeros((H,W), dtype=np.uint8)
        cluster_map = np.full((H,W), -1, dtype=np.int32)
        cluster_vis = np.zeros((H,W,3), dtype=np.uint8)
        return empty_mask, cluster_map, [0,0,0,0], 0.0, cluster_vis, None

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
            # minibatch KMeans: faster for many pixels
            mb = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=n_init)
            mb.fit(lab_scaled)
            labels = mb.predict(lab_scaled)
            centers_scaled = mb.cluster_centers_
        else:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
            labels = km.fit_predict(lab_scaled)
            centers_scaled = km.cluster_centers_
        # recover centers in original LAB scale
        centers_lab = centers_scaled * std.reshape(1,3) + mean.reshape(1,3)
    else:
        # pretrained_centers expected in LAB (k x 3)
        # map each lab_leaf row to nearest center (Euclidean)
        centers_lab = np.asarray(pretrained_centers, dtype=np.float32)
        # scale centers like lab_leaf scaling for distance
        centers_scaled = (centers_lab - mean.reshape(1,3)) / std.reshape(1,3)
        # compute distances
        dif = lab_scaled[:,None,:] - centers_scaled[None,:,:]
        dists = np.linalg.norm(dif, axis=2)  # N x k
        labels = np.argmin(dists, axis=1)

    # build full cluster_map
    cluster_map = np.full((H,W), -1, dtype=np.int32)
    cluster_map[leaf_idx] = labels

    # determine healthy cluster: cluster with largest count
    counts = np.bincount(labels, minlength=k)
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

    # keep largest
    if keep_largest:
        lesion_mask = _keep_largest_component_uint8(lesion_mask, min_area=min_area)

    # bbox + area%
    contours, _ = cv2.findContours(lesion_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        bbox = [0,0,0,0]; area_pct = 0.0
    else:
        total_area = 0.0; xs=[]; ys=[]; xe=[]; ye=[]
        for c in contours:
            x,y,wc,hc = cv2.boundingRect(c)
            xs.append(x); ys.append(y); xe.append(x+wc); ye.append(y+hc)
            total_area += cv2.contourArea(c)
        bbox = [int(min(xs)), int(min(ys)), int(max(xe)-min(xs)), int(max(ye)-min(ys))]
        area_pct = 100.0 * (total_area / (H * W))

    # cluster visualization: map each cluster to its center color (LAB->RGB)
    cluster_vis = np.zeros((H,W,3), dtype=np.uint8)
    if centers_lab is not None:
        # convert centers to uint8 LAB and then to RGB
        centers_lab_u8 = np.clip(centers_lab, 0, 255).astype(np.uint8).reshape(-1,1,3)
        centers_rgb = cv2.cvtColor(centers_lab_u8, cv2.COLOR_LAB2RGB).reshape(-1,3)
        for cid in range(len(centers_rgb)):
            cluster_vis[cluster_map == cid] = centers_rgb[cid]
    # fallback: show original for non-leaf
    cluster_vis[leaf_mask_uint8 == 0] = orig_rgb_np[leaf_mask_uint8 == 0] if 'orig_rgb_np' in globals() else 0

    return lesion_mask, cluster_map, bbox, area_pct, cluster_vis, centers_lab
