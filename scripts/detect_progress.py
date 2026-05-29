"""
detect_progress.py
------------------
Detecta qué imágenes del subset ya fueron revisadas manualmente,
comparando los .txt en data/labeling_subset/ con los originales
en data/labeling/ (que aún tienen las etiquetas auto-generadas por SAM).

Una imagen se considera "revisada" si su .txt en el subset es DISTINTO
del .txt original en labeling/.

Genera reporte y opcionalmente regenera lote para colaborador con solo
las imágenes NO revisadas.

Uso:
    # Solo reportar
    python scripts/detect_progress.py

    # Reportar + regenerar lote para colaborador con las pendientes
    python scripts/detect_progress.py --regenerate-collab

    # Especificar rutas
    python scripts/detect_progress.py --subset data/labeling_subset --original data/labeling
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", type=str, default="data/labeling_subset",
                   help="Carpeta con el subset (donde se etiquetó manualmente)")
    p.add_argument("--original", type=str, default="data/labeling",
                   help="Carpeta original con etiquetas SAM intactas")
    p.add_argument("--regenerate-collab", action="store_true",
                   help="Regenera data/labeling_collab/lote_companero con las pendientes")
    p.add_argument("--collab-dir", type=str, default="data/labeling_collab",
                   help="Carpeta raíz para lotes de colaboración")
    p.add_argument("--clean-old-lotes", action="store_true",
                   help="Elimina lote_1/lote_2 anteriores antes de regenerar (CON CUIDADO)")
    return p.parse_args()


def count_boxes(txt: Path) -> int:
    if not txt.exists():
        return 0
    content = txt.read_text(encoding="utf-8").strip()
    return len([l for l in content.splitlines() if l.strip()])


def main() -> None:
    args = parse_args()
    subset = Path(args.subset)
    original = Path(args.original)

    if not subset.exists():
        raise SystemExit(f"[ERROR] No existe: {subset}")
    if not original.exists():
        raise SystemExit(f"[ERROR] No existe: {original}")

    images = sorted([p for p in subset.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise SystemExit(f"[ERROR] No hay imagenes en {subset}")

    revisadas: list[tuple[Path, int, int]] = []  # (img, n_original, n_actual)
    pendientes: list[Path] = []
    sin_original: list[Path] = []

    for img in images:
        txt_subset = img.with_suffix(".txt")
        txt_original = original / img.with_suffix(".txt").name

        if not txt_original.exists():
            sin_original.append(img)
            continue

        if not txt_subset.exists():
            # No tiene .txt en subset, lo tratamos como pendiente
            pendientes.append(img)
            continue

        new_content = txt_subset.read_text(encoding="utf-8")
        old_content = txt_original.read_text(encoding="utf-8")

        if new_content != old_content:
            revisadas.append((img, count_boxes(txt_original), count_boxes(txt_subset)))
        else:
            pendientes.append(img)

    print(f"Total imágenes en subset:  {len(images)}")
    print(f"  Revisadas (cambiadas):   {len(revisadas)}")
    print(f"  Pendientes (sin tocar):  {len(pendientes)}")
    if sin_original:
        print(f"  Sin contraparte en {original.name}: {len(sin_original)}")

    print(f"\n=== Detalle de revisadas (cajas antes -> después) ===")
    total_orig = 0
    total_new = 0
    for img, n_old, n_new in revisadas[:80]:  # mostrar primeras 80
        delta = n_new - n_old
        sign = "+" if delta >= 0 else ""
        print(f"  {img.name:<50} {n_old:>4} -> {n_new:>4}  ({sign}{delta})")
        total_orig += n_old
        total_new += n_new
    if len(revisadas) > 80:
        print(f"  ... y {len(revisadas) - 80} más")

    if revisadas:
        # Suma sobre TODAS (no solo las 80 mostradas)
        total_orig = sum(n_old for _, n_old, _ in revisadas)
        total_new = sum(n_new for _, _, n_new in revisadas)
        print(f"\n  Total cajas en revisadas: {total_orig} (SAM) -> {total_new} (manual)")
        print(f"  Cambio neto: {total_new - total_orig:+d} cajas")

    if args.regenerate_collab:
        collab_dir = Path(args.collab_dir)

        if args.clean_old_lotes:
            for old in collab_dir.glob("lote_*"):
                if old.is_dir():
                    shutil.rmtree(old)
                    print(f"\n  [LIMPIO] {old.name}")

        nuevo_lote = collab_dir / "lote_companero"
        nuevo_lote.mkdir(parents=True, exist_ok=True)

        for img in pendientes:
            shutil.copy2(str(img), str(nuevo_lote / img.name))
            txt = img.with_suffix(".txt")
            if txt.exists():
                shutil.copy2(str(txt), str(nuevo_lote / txt.name))

        classes_src = subset / "classes.txt"
        if classes_src.exists():
            shutil.copy2(str(classes_src), str(nuevo_lote / "classes.txt"))
        else:
            (nuevo_lote / "classes.txt").write_text("alevin\n", encoding="utf-8")

        print(f"\nLote generado: {nuevo_lote.resolve()}")
        print(f"  {len(pendientes)} imágenes pendientes copiadas")
        print(f"\nAhora ejecuta:")
        print(f"  python scripts/prepare_collab_setup.py")
        print(f"para añadir setup.bat + parches + INSTRUCCIONES en el lote.")


if __name__ == "__main__":
    main()
