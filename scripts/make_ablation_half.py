"""
make_ablation_half.py
---------------------
Crea un dataset "mitad de train" para la ablación controlada de cantidad de
datos: mismo VAL y TEST que el dataset completo, pero con solo ~50% de las
imágenes de entrenamiento (submuestreo contiguo por fuente para preservar la
distribución de videos/fotos y evitar leakage).

Así se compara de forma justa "menos datos vs más datos" sobre el MISMO test.

Salida: data/dataset_yolo_half/ + data_half.yaml
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

SRC = Path("data/dataset_yolo")
DST = Path("data/dataset_yolo_half")
FRACTION = 0.5


def source_of(name: str) -> str:
    m = re.match(r"(VID_\d+_\d+)", name)
    return m.group(1) if m else "foto"


def frame_index(name: str) -> int:
    m = re.search(r"_frame_(\d+)", name)
    return int(m.group(1)) if m else 0


def main() -> None:
    # Limpiar destino
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = DST / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    # val y test: copiar tal cual (idénticos al dataset completo)
    for split in ("val", "test"):
        for img in (SRC / "images" / split).glob("*.jpg"):
            shutil.copy2(img, DST / "images" / split / img.name)
            lbl = SRC / "labels" / split / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, DST / "labels" / split / lbl.name)

    # train: submuestrear ~50% contiguo por fuente
    train_imgs = sorted((SRC / "images" / "train").glob("*.jpg"))
    groups: dict[str, list[Path]] = {}
    for img in train_imgs:
        groups.setdefault(source_of(img.name), []).append(img)

    kept = 0
    for src, imgs in groups.items():
        imgs_sorted = sorted(imgs, key=lambda p: (frame_index(p.name), p.name))
        n_keep = max(1, round(len(imgs_sorted) * FRACTION))
        for img in imgs_sorted[:n_keep]:
            shutil.copy2(img, DST / "images" / "train" / img.name)
            lbl = SRC / "labels" / "train" / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, DST / "labels" / "train" / lbl.name)
            kept += 1

    # yaml
    yaml = DST.parent / "data_half.yaml"
    yaml.write_text(
        f"path: {DST.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames:\n  0: alevin\n", encoding="utf-8")

    n_val = len(list((DST / "images" / "val").glob("*.jpg")))
    n_test = len(list((DST / "images" / "test").glob("*.jpg")))
    print(f"Dataset ablación (mitad de train): train={kept} val={n_val} test={n_test}")
    print(f"yaml: {yaml}")


if __name__ == "__main__":
    main()
