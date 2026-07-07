"""
build_baseline_dataset.py
-------------------------
Construye el dataset YOLO del baseline usando SOLO las imágenes que fueron
revisadas manualmente (cuyo .txt en el subset difiere del auto-generado por SAM
en data/labeling/), y hace un split POR FUENTE/TEMPORAL para evitar data leakage
entre frames casi idénticos de un mismo video.

Estrategia de split (clave metodológica de la tesis):
  - Cada imagen se agrupa por su fuente:
        VID_<fecha>_<hora>  -> frames de ese video
        foto                -> fotos sueltas (IMG_... o timestamps)
  - Dentro de cada grupo se ordena por índice de frame / nombre y se asignan
    BLOQUES CONTIGUOS: primeros 80% -> train, siguiente 10% -> val, últimos 10% -> test.
    Así frames vecinos (casi gemelos) NO caen a la vez en train y test.

Salidas:
  data/dataset_yolo/images/{train,val,test}/
  data/dataset_yolo/labels/{train,val,test}/
  data/counts/conteo_manual.csv       (todas las revisadas: image_name, real_count, split, source)
  data/counts/conteo_manual_test.csv  (solo test: image_name, real_count)  <- para evaluate_counting

Uso:
  python scripts/build_baseline_dataset.py
  python scripts/build_baseline_dataset.py --train 0.7 --val 0.15 --test 0.15
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", type=str, default="data/labeling_subset",
                   help="Carpeta con imágenes + etiquetas revisadas")
    p.add_argument("--original", type=str, default="data/labeling",
                   help="Carpeta con etiquetas auto-generadas (SAM) intactas, para detectar revisadas")
    p.add_argument("--dataset-root", type=str, default="data/dataset_yolo")
    p.add_argument("--counts-dir", type=str, default="data/counts")
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--include-unreviewed", action="store_true",
                   help="Incluir también imágenes NO revisadas (NO recomendado para baseline)")
    return p.parse_args()


def source_of(name: str) -> str:
    """Agrupa por video de origen; las fotos sueltas van a un grupo 'foto'."""
    m = re.match(r"(VID_\d+_\d+)", name)
    if m:
        return m.group(1)
    return "foto"


def frame_index(name: str) -> int:
    """Índice temporal para ordenar frames de un mismo video."""
    m = re.search(r"_frame_(\d+)", name)
    return int(m.group(1)) if m else 0


def count_boxes(txt: Path) -> int:
    if not txt.exists():
        return 0
    content = txt.read_text(encoding="utf-8").strip()
    return len([l for l in content.splitlines() if l.strip()])


def is_reviewed(txt_subset: Path, txt_original: Path) -> bool:
    if not txt_subset.exists():
        return False
    if not txt_original.exists():
        # Sin contraparte original: lo tratamos como revisado si tiene contenido
        return count_boxes(txt_subset) > 0
    return txt_subset.read_text(encoding="utf-8") != txt_original.read_text(encoding="utf-8")


def contiguous_split(items: list, tr: float, va: float) -> tuple[list, list, list]:
    """Divide una lista YA ORDENADA en bloques contiguos train/val/test."""
    n = len(items)
    n_tr = int(round(n * tr))
    n_va = int(round(n * va))
    # Garantizar al menos 1 en val y test si el grupo tiene >=3 elementos
    if n >= 3:
        n_tr = min(n_tr, n - 2)
        n_va = max(n_va, 1)
    train = items[:n_tr]
    val = items[n_tr:n_tr + n_va]
    test = items[n_tr + n_va:]
    return train, val, test


def main() -> None:
    args = parse_args()
    subset = Path(args.subset)
    original = Path(args.original)
    root = Path(args.dataset_root)
    counts_dir = Path(args.counts_dir)

    if not subset.exists():
        raise SystemExit(f"[ERROR] No existe {subset}")

    images = sorted([p for p in subset.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise SystemExit(f"[ERROR] No hay imágenes en {subset}")

    # 1) Seleccionar revisadas y agrupar por fuente
    groups: dict[str, list[Path]] = {}
    n_total = n_revisadas = 0
    for img in images:
        n_total += 1
        txt_sub = img.with_suffix(".txt")
        txt_ori = original / f"{img.stem}.txt"
        reviewed = is_reviewed(txt_sub, txt_ori)
        if not reviewed and not args.include_unreviewed:
            continue
        if not txt_sub.exists():
            continue
        n_revisadas += 1
        groups.setdefault(source_of(img.name), []).append(img)

    if n_revisadas == 0:
        raise SystemExit("[ERROR] No se encontró ninguna imagen revisada.")

    # 2) Limpiar splits previos
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = root / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    # 3) Split contiguo por grupo
    assignments: list[tuple[Path, str, str]] = []  # (img, split, source)
    for src, imgs in groups.items():
        imgs_sorted = sorted(imgs, key=lambda p: (frame_index(p.name), p.name))
        tr, va, te = contiguous_split(imgs_sorted, args.train, args.val)
        for split_name, chunk in (("train", tr), ("val", va), ("test", te)):
            for img in chunk:
                assignments.append((img, split_name, src))

    # 4) Copiar archivos + acumular conteos
    counts_dir.mkdir(parents=True, exist_ok=True)
    manual_rows = []
    split_tally = {"train": 0, "val": 0, "test": 0}
    for img, split_name, src in assignments:
        txt = img.with_suffix(".txt")
        shutil.copy2(str(img), str(root / "images" / split_name / img.name))
        shutil.copy2(str(txt), str(root / "labels" / split_name / txt.name))
        split_tally[split_name] += 1
        manual_rows.append((img.name, count_boxes(txt), split_name, src))

    # 5) Escribir CSVs de conteo (ground truth = nº de cajas revisadas)
    manual_csv = counts_dir / "conteo_manual.csv"
    with manual_csv.open("w", encoding="utf-8") as f:
        f.write("image_name,real_count,split,source\n")
        for name, cnt, split_name, src in manual_rows:
            f.write(f"{name},{cnt},{split_name},{src}\n")

    test_csv = counts_dir / "conteo_manual_test.csv"
    with test_csv.open("w", encoding="utf-8") as f:
        f.write("image_name,real_count\n")
        for name, cnt, split_name, src in manual_rows:
            if split_name == "test":
                f.write(f"{name},{cnt}\n")

    # 6) Reporte
    total_boxes = sum(r[1] for r in manual_rows)
    print("=" * 60)
    print("DATASET BASELINE CONSTRUIDO (solo imágenes revisadas)")
    print("=" * 60)
    print(f"Imágenes totales en subset : {n_total}")
    print(f"Imágenes revisadas usadas  : {n_revisadas}")
    print(f"Alevines etiquetados (cajas): {total_boxes}")
    print(f"\nSplit por fuente (contiguo/temporal):")
    for src, imgs in sorted(groups.items()):
        n = len(imgs)
        print(f"  {src:<28} {n} imágenes")
    print(f"\nReparto final:")
    print(f"  train: {split_tally['train']}  |  val: {split_tally['val']}  |  test: {split_tally['test']}")
    print(f"\nGround truth de conteo:")
    print(f"  {manual_csv}")
    print(f"  {test_csv}  (para evaluate_counting)")
    print(f"\nListo. Siguiente paso: entrenar con train_yolo.py")


if __name__ == "__main__":
    main()
