"""
deduplicate_images.py
---------------------
Detecta imágenes perceptualmente duplicadas en una carpeta y las separa
(no las borra) en una subcarpeta de revisión.

Útil cuando las fotos vienen de ráfagas del celular (3-5 disparos en el mismo
segundo) o de timestamps muy cercanos.

Estrategia:
    1. Calcula pHash de cada imagen.
    2. Agrupa imágenes con distancia Hamming < threshold.
    3. Por cada grupo, mantiene la imagen de mayor resolución/tamaño y mueve
       las demás a 'duplicates/' dentro de la misma carpeta.
    4. Genera un CSV con los grupos para que el usuario pueda revisar.

Uso:
    python scripts/deduplicate_images.py
    python scripts/deduplicate_images.py --src data/raw/images --threshold 5
    python scripts/deduplicate_images.py --src data/raw/images --threshold 5 --dry-run
    python scripts/deduplicate_images.py --src data/raw/images --threshold 5 --apply

Por defecto corre en --dry-run (solo reporta, no mueve).
Para mover realmente los duplicados a 'duplicates/', usa --apply.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image
import imagehash
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deduplica imágenes por pHash.")
    p.add_argument("--src", type=str, default="data/raw/images",
                   help="Carpeta con las imágenes")
    p.add_argument("--threshold", type=int, default=5,
                   help="Distancia Hamming máxima para considerar duplicado (default: 5)")
    p.add_argument("--report-csv", type=str, default="reports/duplicates_report.csv",
                   help="CSV de salida con los grupos detectados")
    p.add_argument("--duplicates-subdir", type=str, default="duplicates",
                   help="Subcarpeta donde mover los duplicados (dentro de --src)")
    p.add_argument("--apply", action="store_true",
                   help="Mover realmente los duplicados (sin esto solo reporta)")
    p.add_argument("--dry-run", action="store_true",
                   help="Solo reportar, no mover (default si no se pasa --apply)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_changes = args.apply and not args.dry_run

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"[ERROR] No existe: {src}")

    images = sorted([p for p in src.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise SystemExit(f"[ERROR] No hay imágenes en {src}")

    print(f"Procesando {len(images)} imágenes en {src}")
    print(f"Threshold de Hamming: {args.threshold}")
    print(f"Modo: {'APPLY (mover archivos)' if apply_changes else 'DRY-RUN (solo reportar)'}\n")

    # Paso 1: calcular pHash de todas las imágenes
    hashes: list[tuple[Path, imagehash.ImageHash, int]] = []
    for img_path in tqdm(images, desc="Calculando pHash", unit="img"):
        try:
            with Image.open(img_path) as im:
                h = imagehash.phash(im)
                size_bytes = img_path.stat().st_size
                hashes.append((img_path, h, size_bytes))
        except Exception as e:
            print(f"[AVISO] No se pudo procesar {img_path.name}: {e}")

    # Paso 2: agrupar por similitud (algoritmo simple O(n^2), suficiente para <5000 imgs)
    groups: list[list[tuple[Path, int]]] = []
    assigned = set()

    for i, (path_i, hash_i, size_i) in enumerate(hashes):
        if i in assigned:
            continue
        group = [(path_i, size_i)]
        assigned.add(i)
        for j in range(i + 1, len(hashes)):
            if j in assigned:
                continue
            path_j, hash_j, size_j = hashes[j]
            if abs(hash_i - hash_j) <= args.threshold:
                group.append((path_j, size_j))
                assigned.add(j)
        if len(group) > 1:
            groups.append(group)

    if not groups:
        print("\nNo se detectaron duplicados perceptuales con ese threshold.")
        return

    print(f"\nGrupos de duplicados detectados: {len(groups)}")
    total_duplicates = sum(len(g) - 1 for g in groups)
    print(f"Imágenes duplicadas totales (a mover): {total_duplicates}")
    print(f"Imágenes únicas tras dedup: {len(images) - total_duplicates}\n")

    # Paso 3: por grupo, conservar la de mayor tamaño y mover el resto
    duplicates_dir = src / args.duplicates_subdir
    if apply_changes:
        duplicates_dir.mkdir(exist_ok=True)

    rows = []
    for gi, group in enumerate(groups, start=1):
        # Ordenar por tamaño descendente -> el "ganador" (mayor) queda como original
        group_sorted = sorted(group, key=lambda x: -x[1])
        winner, *losers = group_sorted

        rows.append({
            "group_id": gi,
            "role": "keep",
            "filename": winner[0].name,
            "size_bytes": winner[1],
        })
        for loser_path, loser_size in losers:
            rows.append({
                "group_id": gi,
                "role": "duplicate",
                "filename": loser_path.name,
                "size_bytes": loser_size,
            })
            if apply_changes:
                target = duplicates_dir / loser_path.name
                # Evitar colisiones de nombre
                if target.exists():
                    target = duplicates_dir / f"{loser_path.stem}_{gi}{loser_path.suffix}"
                shutil.move(str(loser_path), str(target))

    # Paso 4: guardar CSV
    report_csv = Path(args.report_csv)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "role", "filename", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Reporte guardado en: {report_csv}")
    if apply_changes:
        print(f"Duplicados movidos a: {duplicates_dir}")
        print("Revisa esa carpeta para confirmar antes de eliminarlos definitivamente.")
    else:
        print("\nFue DRY-RUN. Para mover realmente los duplicados ejecuta:")
        print(f"  python scripts/deduplicate_images.py --src {args.src} --threshold {args.threshold} --apply")


if __name__ == "__main__":
    main()
