"""
"""

import numpy as np
from typing import Literal
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, remove_small_holes, closing, square
from skimage.exposure import rescale_intensity

def global_thresholding(
    img: np.ndarray,
    method: Literal["otsu", "manual"] = "otsu",
    t: float | None = None,
    invert: bool = False,
    gaussian_sigma: float = 0.0,
    cleanup: bool = True,
    min_area: int = 64
) -> np.ndarray:
    """
    Segmentação por limiarização global.
    - Converte RGB->gray se necessário.
    - Usa Otsu por padrão; também aceita limiar manual (0..1).
    - Opcional: suavização Gaussiana e pós-processamento morfológico.

    Retorna:
        mask (np.ndarray bool) com shape (H, W)

    Parâmetros principais:
      method: "otsu" | "manual"
      t: limiar global (0..1) se method="manual"
      invert: inverte a máscara no final (útil se o objeto é mais escuro)
      gaussian_sigma: suavização antes do limiar (ex.: 1.0)
      cleanup: aplica closing + remove ruído pequeno
      min_area: tamanho mínimo de componente conectado a manter (px)
    """
    # 1) Garantir escala de cinza float [0,1]
    if img.ndim == 3:
        gray = rgb2gray(img)  # float64 [0,1]
    else:
        gray = img.astype(np.float32, copy=False)

    # Normaliza para evitar faixas estranhas
    gray = rescale_intensity(gray, in_range="image", out_range=(0.0, 1.0))

    # 2) Suavização opcional (estabiliza Otsu em ruído)
    if gaussian_sigma and gaussian_sigma > 0:
        gray = gaussian(gray, sigma=gaussian_sigma, preserve_range=True)

    # 3) Limiar
    if method == "otsu":
        t_val = float(threshold_otsu(gray))
    elif method == "manual":
        if t is None:
            raise ValueError("For method='manual', provide t in [0,1].")
        t_val = float(t)
    else:
        raise ValueError("method must be 'otsu' or 'manual'.")

    mask = gray > t_val

    # 4) Pós-processamento opcional (remove ruído, fecha buracos)
    if cleanup:
        # fecha pequenos “furos” nas bordas
        mask = closing(mask, square(3))
        # remove componentes pequenas
        mask = remove_small_objects(mask, min_size=max(1, int(min_area)))
        # preenche pequenos buracos
        mask = remove_small_holes(mask, area_threshold=max(1, int(min_area)))

    # 5) Inversão se desejado (útil quando objeto é o mais escuro)
    if invert:
        mask = ~mask

    return mask

# This is NOT a script file.
if __name__ == '__main__':
    raise RuntimeError("This module is not a standalone script.")