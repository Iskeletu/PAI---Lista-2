"""
Fábio Gandini - 04/11/2025

Implementação técnica das questões 1 e 2 da Lista 2 de Processamento e Analise de Imagens.
"""

# Native Modules:
from typing import Final
from pathlib import Path
import time
import csv

# Internal Modules:
from global_thresholding import global_thresholding
from k_means import k_means

#External Modules:
from skimage import io, img_as_float32, img_as_ubyte
from skimage.color import rgb2gray
import numpy as np

# Constants:
IMAGE_FOLDER_NAME: Final[str] = "img"
IMAGE_DIRECTORY: Final[Path] = (Path(__file__).parent.parent / IMAGE_FOLDER_NAME).resolve()
VALID_FILE_EXTENSIONS:Final[frozenset[str]] = frozenset({".png", ".jpg", ".jpeg"})

OUTPUT_FOLDER_NAME: Final[str] = "output"
OUTPUT_DIRECTORY: Final[Path] = (Path(__file__).parent.parent / OUTPUT_FOLDER_NAME).resolve()


def open_images(convert_to_grayscale:bool) -> list[dict]:
    """
    """

    images:list[dict] = []
    
    if not IMAGE_DIRECTORY.exists():
        raise FileNotFoundError(f"Folder not found: {IMAGE_DIRECTORY}")

    files = [file for file in IMAGE_DIRECTORY.iterdir() if file.is_file() and file.suffix.lower() in VALID_FILE_EXTENSIONS]
    files = sorted(files, key=lambda f: f.name.lower()) # Sorts files by name.

    for file in files:
        try:
            img = io.imread(file)
            if convert_to_grayscale and img.ndim == 3:
                img = rgb2gray(img)
            img = img_as_float32(img)
            images.append({
                "name": file.stem,
                "path": str(file),
                "image": img
            })
        except Exception as e:
            print(f"(Skipping) Failure to read {file.name}: {e}")

    if not images:
        raise FileNotFoundError(f"No valid images found in: {IMAGE_DIRECTORY}.")

    return images






def save_mask(mask: np.ndarray, path: Path) -> None:
    # salva bool como 0/255 (PNG)
    io.imsave(path.as_posix(), img_as_ubyte(mask), check_contrast=False)

def make_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # overlay simples: pinta pixels do mask em vermelho
    if img.ndim == 2:
        # cinza -> RGB
        img_rgb = np.stack([img, img, img], axis=-1)
    else:
        img_rgb = img.copy()
    ov = img_rgb.copy()
    ov[..., 0] = np.clip(ov[..., 0] + 0.6*mask, 0, 1)  # canal R
    return ov

def foreground_ratio(mask: np.ndarray) -> float:
    return float(mask.sum()) / mask.size


def run_and_save_all():
    imgs = open_images(convert_to_grayscale=False)  # mantenha RGB; Otsu usa gray internamente
    metrics_csv = OUTPUT_DIRECTORY / "metrics.csv"

    # cabeçalho do CSV
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "method", "fg_ratio", "time_ms", "notes"])

    for item in imgs:
        name = item["name"]
        img  = item["image"]

        # --- OTSU / Global Thresholding ---
        t0 = time.perf_counter()
        # se seu global_thresholding espera gray, converta dentro da função (ou aqui)
        m_otsu = global_thresholding(img)  
        dt_otsu = (time.perf_counter() - t0) * 1000

        save_mask(m_otsu, OUTPUT_DIRECTORY / "otsu" / f"{name}_otsu.png")
        ov_otsu = make_overlay(img, m_otsu)
        io.imsave((OUTPUT_DIRECTORY / "overlay" / f"{name}_overlay_otsu.png").as_posix(),
                  img_as_ubyte(ov_otsu), check_contrast=False)

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([name, "otsu",
                                    f"{foreground_ratio(m_otsu):.4f}",
                                    f"{dt_otsu:.1f}",
                                    ""])

        # --- K-MEANS (intensidade) ---
        t1 = time.perf_counter()
        m_km = k_means(img, k=2, use_color=False, add_spatial=True, spatial_weight=0.2, seed=0)
        dt_km = (time.perf_counter() - t1) * 1000

        save_mask(m_km, OUTPUT_DIRECTORY / "kmeans" / f"{name}_kmeans.png")
        ov_km = make_overlay(img, m_km)
        io.imsave((OUTPUT_DIRECTORY / "overlay" / f"{name}_overlay_kmeans.png").as_posix(),
                  img_as_ubyte(ov_km), check_contrast=False)

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([name, "kmeans",
                                    f"{foreground_ratio(m_km):.4f}",
                                    f"{dt_km:.1f}",
                                    "use_color=False, spatial=0.2"])


def main():
    try:
        run_and_save_all()
        print(f"\n✅ Resultados salvos em: {OUTPUT_DIRECTORY}")
        print(f"   - Máscaras: {OUTPUT_DIRECTORY/'otsu'} e {OUTPUT_DIRECTORY/'kmeans'}")
        print(f"   - Overlays: {OUTPUT_DIRECTORY/'overlay'}")
        print(f"   - Métricas: {OUTPUT_DIRECTORY/'metrics.csv'}")
    except FileNotFoundError as e:
        print(f"Erro: {e}")
    except Exception as e:
        print(f"Falha inesperada: {e}")


# This is a script file and should NOT be imported:
if __name__ == '__main__':
    IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "otsu").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "kmeans").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "overlay").mkdir(parents=True, exist_ok=True)

    main()