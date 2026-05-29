"""
merge_collab.py
---------------
Unifica las etiquetas devueltas por un colaborador (carpeta lote_N revisada)
con el subset principal data/labeling_subset/.

Compara modos:
    - Si el .txt en el lote devuelto es DISTINTO del original, se considera
      revisado y se copia al subset principal.
    - Si el .txt es identico al original, se asume que el colaborador no
      lo toco (puede haber saltado esa imagen).

Genera un reporte con:
    - Cuantas etiquetas se actualizaron
    - Cuantas quedaron sin tocar
    - Imagenes que el colaborador "marco como revisadas" (cambiaron)

Uso:
    python scripts/merge_collab.py path/al/lote_devuelto
    python scripts/merge_collab.py path/al/lote_devuelto --dst data/labeling_subset
    python scripts/merge_collab.py path/al/lote_devuelto --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unifica lote etiquetado al subset principal.")
    p.add_argument("lote_dir", type=str,
                   help="Carpeta con las imagenes+.txt devueltos por el colaborador")
    p.add_argument("--dst", type=str, default="data/labeling_subset",
                   help="Carpeta destino (subset principal)")
    p.add_argument("--dry-run", action="store_true",
                   help="Solo reportar, no modificar destino")
    p.add_argument("--backup-dir", type=str, default="data/labeling_subset_backup",
                   help="Carpeta donde guardar backup de los .txt sobrescritos")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lote_dir = Path(args.lote_dir)
    dst = Path(args.dst)
    backup_dir = Path(args.backup_dir)

    if not lote_dir.exists():
        raise SystemExit(f"[ERROR] No existe: {lote_dir}")
    if not dst.exists():
        raise SystemExit(f"[ERROR] No existe destino: {dst}")

    # Buscar los .txt en el lote devuelto
    lote_txts = sorted([p for p in lote_dir.iterdir()
                        if p.is_file() and p.suffix == ".txt" and p.name != "classes.txt"])

    if not lote_txts:
        raise SystemExit(f"[ERROR] No hay .txt en {lote_dir}")

    print(f"Lote devuelto: {lote_dir.resolve()}")
    print(f"Destino:       {dst.resolve()}")
    print(f"Modo:          {'DRY-RUN' if args.dry_run else 'APLICAR'}")
    print(f"Backup en:     {backup_dir.resolve()}\n")

    updated = 0
    identical = 0
    missing_in_dst = 0
    backed_up = 0

    if not args.dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)

    for lote_txt in lote_txts:
        dst_txt = dst / lote_txt.name
        if not dst_txt.exists():
            missing_in_dst += 1
            print(f"  [missing] {lote_txt.name} no existe en destino, omitida")
            continue

        new_content = lote_txt.read_text(encoding="utf-8")
        old_content = dst_txt.read_text(encoding="utf-8")

        if new_content == old_content:
            identical += 1
            continue

        updated += 1
        # Conteo de cajas para info
        old_n = len([l for l in old_content.splitlines() if l.strip()])
        new_n = len([l for l in new_content.splitlines() if l.strip()])
        delta = new_n - old_n
        delta_str = f"+{delta}" if delta >= 0 else f"{delta}"
        print(f"  [updated] {lote_txt.name}: {old_n} -> {new_n} cajas ({delta_str})")

        if not args.dry_run:
            shutil.copy2(str(dst_txt), str(backup_dir / dst_txt.name))
            backed_up += 1
            shutil.copy2(str(lote_txt), str(dst_txt))

    print(f"\n=== Resumen ===")
    print(f"Archivos en lote devuelto:    {len(lote_txts)}")
    print(f"Etiquetas actualizadas:       {updated}")
    print(f"Etiquetas identicas (sin cambio): {identical}")
    print(f"No encontradas en destino:    {missing_in_dst}")
    if not args.dry_run:
        print(f"Backups guardados:            {backed_up} en {backup_dir}")
    else:
        print("(no se modifico nada, ejecuta sin --dry-run para aplicar)")


if __name__ == "__main__":
    main()
