"""
split_dataset.py
----------------
Divide un conjunto de imágenes etiquetadas en train/val/test
con la estructura que YOLO espera:

    data/dataset_yolo/
        images/{train,val,test}/
        labels/{train,val,test}/

Espera que las imágenes y sus .txt YOLO estén juntos en una misma carpeta
de origen (por defecto data/frames/selected/), o las imágenes en una carpeta
y las etiquetas en otra (--labels-src).

Uso:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --train 0.8 --val 0.1 --test 0.1
    python scripts/split_dataset.py --images-src data/frames/selected --labels-src data/frames/selected
    python scripts/split_dataset.py --seed 42 --move        # mueve en vez de copiar
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Divide dataset YOLO en train/val/test.")
    parser.add_argument("--images-src", type=str, default="data/frames/selected",
                        help="Carpeta con las imágenes etiquetadas")
    parser.add_argument("--labels-src", type=str, default=None,
                        help="Carpeta con las etiquetas .txt. Si se omite, se asume la misma que --images-src")
    parser.add_argument("--dataset-root", type=str, default="data/dataset_yolo",
                        help="Raíz del dataset YOLO de salida")
    parser.add_argument("--train", type=float, default=0.8, help="Proporción train (default: 0.8)")
    parser.add_argument("--val", type=float, default=0.1, help="Proporción val (default: 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="Proporción test (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (default: 42)")
    parser.add_argument("--move", action="store_true",
                        help="Mover archivos en vez de copiarlos")
    parser.add_argument("--allow-unlabeled", action="store_true",
                        help="Permitir imágenes sin .txt (default: se omiten)")
    return parser.parse_args()


def validate_split(train: float, val: float, test: float) -> None:
    total = train + val + test
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise SystemExit(f"[ERROR] La suma de train+val+test debe ser 1.0 (es {total})")


def find_pairs(images_src: Path, labels_src: Path, allow_unlabeled: bool) -> list[tuple[Path, Path | None]]:
    """Devuelve lista de (imagen, label .txt o None)."""
    images = sorted([p for p in images_src.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    pairs: list[tuple[Path, Path | None]] = []
    missing = 0
    for img in images:
        lbl = labels_src / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            missing += 1
            if allow_unlabeled:
                pairs.append((img, None))
    if missing:
        msg = f"[AVISO] {missing} imágenes sin .txt"
        msg += " (incluidas como negativos)" if allow_unlabeled else " (omitidas)"
        print(msg)
    return pairs


def transfer(src: Path, dst: Path, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main() -> None:
    args = parse_args()
    validate_split(args.train, args.val, args.test)

    images_src = Path(args.images_src)
    labels_src = Path(args.labels_src) if args.labels_src else images_src
    dataset_root = Path(args.dataset_root)

    if not images_src.exists():
        raise SystemExit(f"[ERROR] No existe la carpeta de imágenes: {images_src.resolve()}")
    if not labels_src.exists():
        raise SystemExit(f"[ERROR] No existe la carpeta de etiquetas: {labels_src.resolve()}")

    pairs = find_pairs(images_src, labels_src, args.allow_unlabeled)
    if not pairs:
        raise SystemExit(f"[ERROR] No se encontraron pares imagen/etiqueta en {images_src.resolve()}")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(round(n * args.train))
    n_val = int(round(n * args.val))
    # El test absorbe el resto para asegurar que sumamos exactamente n
    n_test = n - n_train - n_val

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    print(f"Total pares encontrados: {n}")
    print(f"  train: {n_train}  |  val: {n_val}  |  test: {n_test}")
    print(f"Operación: {'MOVER' if args.move else 'COPIAR'}\n")

    for split_name, items in splits.items():
        img_dir = dataset_root / "images" / split_name
        lbl_dir = dataset_root / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img, lbl in items:
            transfer(img, img_dir / img.name, args.move)
            if lbl is not None:
                transfer(lbl, lbl_dir / lbl.name, args.move)

        print(f"  {split_name}: {len(items)} imágenes -> {img_dir.relative_to(dataset_root.parent)}")

    print("\nListo. Recuerda revisar que data/dataset_yolo/data.yaml apunte al dataset.")


if __name__ == "__main__":
    main()
