"""
apply_curation.py
-----------------
Aplica la selección hecha en el visor HTML de curate_frames.py.

Lee selection.json (descargado desde el navegador) y copia los frames
marcados como "kept" a data/frames/selected/.

Uso:
    python scripts/apply_curation.py path/to/selection.json
    python scripts/apply_curation.py path/to/selection.json --move
    python scripts/apply_curation.py path/to/selection.json --src data/frames/all --dst data/frames/selected
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aplica curaduría de frames desde selection.json")
    p.add_argument("selection_json", type=str,
                   help="Archivo selection.json descargado del visor")
    p.add_argument("--src", type=str, default="data/frames/all",
                   help="Carpeta de origen de los frames")
    p.add_argument("--dst", type=str, default="data/frames/selected",
                   help="Carpeta destino para los frames conservados")
    p.add_argument("--move", action="store_true",
                   help="Mover los archivos en vez de copiarlos")
    p.add_argument("--overwrite", action="store_true",
                   help="Sobrescribir archivos existentes en destino")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sel_path = Path(args.selection_json)
    if not sel_path.exists():
        raise SystemExit(f"[ERROR] No existe: {sel_path}")

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"[ERROR] No existe carpeta origen: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    data = json.loads(sel_path.read_text(encoding="utf-8"))
    kept = data.get("kept", [])
    discarded = data.get("discarded", [])
    total = data.get("total", len(kept) + len(discarded))

    print(f"Total frames en selección: {total}")
    print(f"  Conservados (kept):    {len(kept)}")
    print(f"  Descartados:           {len(discarded)}")
    print(f"\nOperación: {'MOVER' if args.move else 'COPIAR'}")
    print(f"  Origen:  {src.resolve()}")
    print(f"  Destino: {dst.resolve()}\n")

    if not kept:
        raise SystemExit("[AVISO] Lista 'kept' vacía. Nada que hacer.")

    copied = 0
    missing = 0
    skipped = 0

    for name in kept:
        src_path = src / name
        dst_path = dst / name

        if not src_path.exists():
            print(f"  [missing] {name}")
            missing += 1
            continue
        if dst_path.exists() and not args.overwrite:
            skipped += 1
            continue

        if args.move:
            shutil.move(str(src_path), str(dst_path))
        else:
            shutil.copy2(str(src_path), str(dst_path))
        copied += 1

    print(f"\nProcesados: {copied} {'movidos' if args.move else 'copiados'}")
    if skipped:
        print(f"Omitidos (ya existían en destino): {skipped}  -- usa --overwrite para forzar")
    if missing:
        print(f"No encontrados en origen: {missing}")


if __name__ == "__main__":
    main()
