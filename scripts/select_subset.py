"""
select_subset.py
----------------
Selecciona un subset estratificado de imágenes para revisión manual
en LabelImg, en base al número de detecciones del auto-label.

La estratificación asegura que el baseline entrenado vea TODOS los
regímenes de densidad (pocos, medios, muchos alevines).

Uso:
    python scripts/select_subset.py
    python scripts/select_subset.py --target-total 150
    python scripts/select_subset.py --src data/labeling --dst data/labeling_subset --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Bins de estratificación (rango_min, rango_max_inclusive, etiqueta)
BINS = [
    (1, 30, "baja"),
    (31, 60, "media_baja"),
    (61, 100, "media_alta"),
    (101, 200, "alta"),
    (201, 10000, "muy_alta"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Selecciona subset estratificado.")
    p.add_argument("--src", type=str, default="data/labeling")
    p.add_argument("--dst", type=str, default="data/labeling_subset")
    p.add_argument("--target-total", type=int, default=150,
                   help="Tamaño objetivo aproximado del subset (default: 150)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy-labels", action="store_true", default=True,
                   help="Copiar también los .txt YOLO (default: True)")
    return p.parse_args()


def count_detections(txt_path: Path) -> int:
    if not txt_path.exists():
        return 0
    content = txt_path.read_text().strip()
    if not content:
        return 0
    return len(content.splitlines())


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"[ERROR] No existe: {src}")

    # Reunir todas las imágenes con su conteo de detecciones
    items: list[tuple[Path, int]] = []
    for img_path in sorted(src.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if img_path.name == "classes.txt":
            continue
        n = count_detections(img_path.with_suffix(".txt"))
        items.append((img_path, n))

    if not items:
        raise SystemExit(f"[ERROR] No hay imágenes en {src}")

    # Clasificar por bin
    by_bin: dict[str, list[tuple[Path, int]]] = {b[2]: [] for b in BINS}
    for path, n in items:
        for lo, hi, label in BINS:
            if lo <= n <= hi:
                by_bin[label].append((path, n))
                break

    print(f"Total imágenes con etiquetas: {len(items)}\n")
    print("Distribución por bin:")
    for lo, hi, label in BINS:
        n_in_bin = len(by_bin[label])
        print(f"  {label:>12}  ({lo:>3}-{hi:>4} det): {n_in_bin:>4} imágenes")

    # Política de muestreo: TODO de bins chicos, sample proporcional de bins grandes
    # Pesos aproximados: balance entre representar la distribución y cobertura
    target = args.target_total
    # Distribución objetivo por bin (porcentajes)
    weights = {
        "baja":        0.25,  # subrepresentado en data, importante para diversidad
        "media_baja":  0.25,
        "media_alta":  0.20,
        "alta":        0.25,
        "muy_alta":    0.05,
    }

    rng = random.Random(args.seed)
    selected: list[Path] = []
    print(f"\nMuestreo (target ~{target}):")
    for lo, hi, label in BINS:
        bucket = by_bin[label]
        if not bucket:
            print(f"  {label:>12}: 0 disponibles")
            continue
        # Cuántos queremos de este bin
        desired = max(1, int(round(target * weights[label])))
        take = min(desired, len(bucket))
        sample = rng.sample(bucket, take) if take < len(bucket) else bucket
        selected.extend([p for p, _ in sample])
        print(f"  {label:>12}: tomar {take:>3} de {len(bucket):>4}")

    print(f"\nSubset final: {len(selected)} imágenes")

    # Copiar a dst
    dst.mkdir(parents=True, exist_ok=True)
    for img_path in selected:
        shutil.copy2(str(img_path), str(dst / img_path.name))
        if args.copy_labels:
            lbl = img_path.with_suffix(".txt")
            if lbl.exists():
                shutil.copy2(str(lbl), str(dst / lbl.name))

    # Copiar classes.txt si existe
    classes_src = src / "classes.txt"
    if classes_src.exists():
        shutil.copy2(str(classes_src), str(dst / "classes.txt"))

    print(f"\nCopiado a: {dst.resolve()}")
    print("\nPróximo paso: revisar en LabelImg con")
    print(f"  labelImg {dst} {dst}/classes.txt {dst}")


if __name__ == "__main__":
    main()
