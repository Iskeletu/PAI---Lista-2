# src/shapes.py
import numpy as np
from skimage.morphology import convex_hull_image
from pathlib import Path
from skimage import io, img_as_ubyte

BASE = Path(__file__).parent.parent
IMG_DIR = BASE / "img"
OUT_DIR = BASE / "output"

def shape_convex_hull(mask: np.ndarray) -> np.ndarray:
    """
    """
    return convex_hull_image(mask)

def _to_rgb01(img):
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    return img.astype(np.float32) if img.dtype != np.float32 else img

def _overlay_mask(img, mask, color=(0.0, 1.0, 0.0), alpha=0.6):
    img = _to_rgb01(img)
    out = img.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=out.dtype)
    return out

def _load_img01(p: Path):
    x = io.imread(p)
    if x.dtype != np.float32 and x.dtype != np.float64:
        x = x.astype(np.float32) / 255.0
    return x

def _load_mask(p: Path):
    m = io.imread(p)
    if m.ndim == 3:
        m = m[..., 0]
    return m > 0  # 0/255 -> bool

def run_hull(name: str, prefer: str = "otsu"):
    """
    """
    (OUT_DIR / "shapes" / "convex_hull").mkdir(parents=True, exist_ok=True)

    # original pode ser .png/.jpg/.jpeg
    original = next((p for ext in (".png", ".jpg", ".jpeg")
                     if (p := (IMG_DIR / f"{name}{ext}")).exists()), None)
    if original is None:
        raise FileNotFoundError(f"Original não encontrado em img/: {name}.*")

    mask_path = OUT_DIR / prefer / f"{name}_{prefer}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Máscara não encontrada: {mask_path}")

    img  = _load_img01(original)
    mask = _load_mask(mask_path)

    hull = shape_convex_hull(mask)
    ov   = _overlay_mask(img, hull, color=(0.0, 1.0, 0.0), alpha=0.6)
    io.imsave((OUT_DIR / "shapes" / "convex_hull" / f"{name}_hull.png").as_posix(),
              img_as_ubyte(ov), check_contrast=False)

    print(f"[Q2] Convex Hull salvo em: out/shapes/convex_hull/{name}_hull.png")

if __name__ == "__main__":
    # exemplo: ajuste para um nome real seu, já segmentado na Q1
    run_hull("example_image_1", prefer="otsu")