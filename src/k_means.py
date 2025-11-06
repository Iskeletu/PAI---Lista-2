"""
Fábio Gandini - 04/11/2025

Implementação técnica do algorítimo do segmentação de imagens K-Means.
"""

#External Modules:
from skimage.color import rgb2lab, rgb2gray
import numpy as np


def k_means(
    img: np.ndarray,
    k: int = 2,
    use_color: bool = False,
    add_spatial: bool = False,
    spatial_weight: float = 0.0,
    max_iter: int = 100,
    n_init: int = 5,
    seed: int = 0,
    return_labels: bool = False,
):
    """
    K-means para segmentação. img deve estar em float [0,1].
    Se use_color=False e img for RGB, converte para gray corretamente.
    """
    h, w = img.shape[:2]

    # --- Features ---
    if img.ndim == 3 and use_color:
        # LAB para usar luminância L (~[0,100]) e cromas a/b
        lab = rgb2lab(img)
        L = (lab[..., 0] / 100.0).astype(np.float32)
        feat = lab.reshape(-1, 3).astype(np.float32)
        lum = L.reshape(-1, 1)  # para decidir "foreground"
    else:
        # Garantir 1 canal quando não usar cor
        if img.ndim == 3:
            gray = rgb2gray(img)  # (H,W) float64 in [0,1]
        else:
            gray = img
        gray = gray.astype(np.float32)
        feat = gray.reshape(-1, 1)  # N = H*W
        lum = feat.copy()

    if add_spatial and spatial_weight > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        yy = (yy.astype(np.float32) / float(h)).reshape(-1, 1)
        xx = (xx.astype(np.float32) / float(w)).reshape(-1, 1)
        feat = np.hstack([feat, spatial_weight * xx, spatial_weight * yy])

    # Sanity: no NaN/Inf
    if not np.isfinite(feat).all():
        raise ValueError("Non-finite values in features. Check input image preprocessing.")

    # --- K-means com n_init ---
    rng = np.random.default_rng(seed)
    best_inertia = np.inf
    best_centers, best_labels = None, None

    for _ in range(n_init):
        # inicialização simples: amostra k pixels
        idx = rng.choice(feat.shape[0], size=k, replace=False)
        centers = feat[idx].copy()

        for _ in range(max_iter):
            # Atribuição
            dists = np.linalg.norm(feat[:, None, :] - centers[None, :, :], axis=2)
            new_labels = dists.argmin(axis=1)

            # Atualização
            new_centers = []
            for i in range(k):
                pts = feat[new_labels == i]
                if pts.size == 0:
                    # evita cluster vazio reamostrando um ponto aleatório
                    new_centers.append(feat[rng.integers(0, feat.shape[0])])
                else:
                    new_centers.append(pts.mean(axis=0))
            new_centers = np.vstack(new_centers).astype(np.float32)

            # Convergência: rótulos não mudaram
            if best_labels is not None and np.array_equal(new_labels, best_labels):
                centers = new_centers
                break

            # ou pouca mudança nos centros
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            best_labels = new_labels
            if shift < 1e-6:
                break

        inertia = ((feat - centers[best_labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers, final_labels = centers.copy(), best_labels.copy()

    labels = final_labels  # shape: (H*W,)
    centers = best_centers

    # Escolhe foreground pelo maior brilho
    cluster_means = np.array([
        lum[labels == i].mean() if np.any(labels == i) else -np.inf for i in range(k)
    ])
    fg_cluster = int(cluster_means.argmax())

    mask = (labels.reshape(h, w) == fg_cluster)

    if return_labels:
        return mask, labels.reshape(h, w), centers
    return mask


# This is NOT a script file.
if __name__ == '__main__':
    raise RuntimeError("This module is not a standalone script.")