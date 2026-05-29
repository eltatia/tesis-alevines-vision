"""
split_for_collab.py
-------------------
Divide el subset de etiquetado en N lotes balanceados por densidad,
para repartir entre colaboradores. Cada lote queda como una carpeta
independiente que se puede enviar por Drive/WeTransfer y abrirse
directamente en LabelImg.

Estructura de salida:
    data/labeling_collab/
        lote_1/
            <imagenes>.jpg
            <etiquetas>.txt
            classes.txt
            launch_labelimg.bat
            INSTRUCCIONES.txt
        lote_2/
            ...

Uso:
    python scripts/split_for_collab.py
    python scripts/split_for_collab.py --n-lotes 2
    python scripts/split_for_collab.py --n-lotes 3 --src data/labeling_subset
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Divide subset en lotes para colaboracion.")
    p.add_argument("--src", type=str, default="data/labeling_subset",
                   help="Carpeta del subset a dividir")
    p.add_argument("--dst", type=str, default="data/labeling_collab",
                   help="Carpeta raiz donde generar los lotes")
    p.add_argument("--n-lotes", type=int, default=2,
                   help="Numero de lotes a generar (default: 2)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def count_detections(txt_path: Path) -> int:
    if not txt_path.exists():
        return 0
    content = txt_path.read_text().strip()
    return len(content.splitlines()) if content else 0


INSTRUCCIONES_TEMPLATE = """\
INSTRUCCIONES PARA ETIQUETADO COLABORATIVO
==========================================

Este lote contiene {n_imagenes} imagenes de alevines con etiquetas
pre-generadas automaticamente (cajas verdes ya dibujadas).

Tu tarea: revisar cada imagen y CORREGIR las etiquetas (no dibujar desde cero).

PASOS:
------
1. Instalar Python 3.10 si no lo tienes:
   https://www.python.org/downloads/

2. Abrir una terminal en esta carpeta y ejecutar:
   pip install labelImg

3. Doble click en launch_labelimg.bat

4. En LabelImg:
   - Cambiar formato a YOLO (boton en sidebar izquierda hasta que diga "YOLO")
   - Activar Auto Save (menu View -> Auto Save mode)

5. Atajos esenciales:
   W = nueva caja        |  D = siguiente imagen
   Ctrl+S = guardar      |  A = imagen anterior
   Del = borrar caja     |  Ctrl+D = duplicar caja

6. Para cada imagen:
   - Borrar cajas falsas (zapatos, marca de bandeja, sombras)
   - Agregar alevines que no estan marcados (W para nueva caja)
   - Ajustar cajas mal posicionadas (arrastrar esquinas)
   - Guardar antes de pasar a la siguiente (auto-save lo hace si esta activo)

CRITERIO DE ETIQUETADO:
-----------------------
- Etiquetar cada alevin visible con mas del 50% del cuerpo en cuadro
- NO etiquetar alevines totalmente ocluidos por otros
- UNA caja por alevin (no por cabeza+cola por separado)
- En cumulos densos, etiquetar solo los que se distinguen claramente

DEVOLUCION:
-----------
Cuando termines (o al hacer pausa para sincronizar):
- Comprimir TODA esta carpeta en un ZIP
- Enviarlo de vuelta por Drive/WeTransfer
- El responsable principal lo procesara con scripts/merge_collab.py

DUDAS:
------
Si la app se cierra al dibujar/hacer zoom, hay un bug conocido con
PyQt5 + Python 3.10. Pide los archivos parcheados de canvas.py,
shape.py y labelImg.py al responsable principal.
"""

LAUNCH_BAT = """\
@echo off
REM Abre LabelImg con esta carpeta ya cargada
cd /d "%~dp0"
labelImg "%~dp0" "%~dp0\\classes.txt" "%~dp0"
"""


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise SystemExit(f"[ERROR] No existe: {src}")
    if args.n_lotes < 2:
        raise SystemExit("[ERROR] --n-lotes debe ser >= 2")

    # Reunir todas las imagenes con su densidad
    items: list[tuple[Path, int]] = []
    for img_path in sorted(src.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        n = count_detections(img_path.with_suffix(".txt"))
        items.append((img_path, n))

    if not items:
        raise SystemExit(f"[ERROR] No hay imagenes en {src}")

    print(f"Total imagenes a repartir: {len(items)}")
    print(f"Numero de lotes: {args.n_lotes}\n")

    # Ordenar por densidad y distribuir round-robin para balancear
    # (cada lote recibe imagenes de todas las densidades)
    items.sort(key=lambda x: x[1])
    rng = random.Random(args.seed)
    # Pequeno shuffle dentro de cada nivel para no agrupar timestamps
    rng.shuffle(items)
    items.sort(key=lambda x: x[1])  # vuelve a ordenar por densidad

    lotes: list[list[Path]] = [[] for _ in range(args.n_lotes)]
    for i, (path, _n) in enumerate(items):
        lotes[i % args.n_lotes].append(path)

    # Crear carpetas y copiar
    for idx, lote in enumerate(lotes, start=1):
        lote_dir = dst / f"lote_{idx}"
        lote_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for img in lote:
            shutil.copy2(str(img), str(lote_dir / img.name))
            lbl = img.with_suffix(".txt")
            if lbl.exists():
                shutil.copy2(str(lbl), str(lote_dir / lbl.name))
            copied += 1

        # Copiar classes.txt
        classes_src = src / "classes.txt"
        if classes_src.exists():
            shutil.copy2(str(classes_src), str(lote_dir / "classes.txt"))
        else:
            (lote_dir / "classes.txt").write_text("alevin\n", encoding="utf-8")

        # Instrucciones + launcher
        (lote_dir / "INSTRUCCIONES.txt").write_text(
            INSTRUCCIONES_TEMPLATE.format(n_imagenes=copied), encoding="utf-8"
        )
        (lote_dir / "launch_labelimg.bat").write_text(LAUNCH_BAT, encoding="utf-8")

        # Reporte de densidad
        densities = [count_detections((lote_dir / img.name).with_suffix(".txt"))
                     for img in lote]
        avg = sum(densities) / len(densities) if densities else 0
        print(f"  lote_{idx}: {copied} imagenes  "
              f"(densidad detectada: min={min(densities) if densities else 0}, "
              f"max={max(densities) if densities else 0}, prom={avg:.0f})")

    print(f"\nLotes generados en: {dst.resolve()}")
    print(f"\nPara enviar a un colaborador:")
    print(f"  1. Comprimir la carpeta lote_N en ZIP")
    print(f"  2. Subir a Drive/WeTransfer/etc")
    print(f"  3. Compartir link con el colaborador")
    print(f"\nCuando devuelva el lote etiquetado, usar:")
    print(f"  python scripts/merge_collab.py <ruta-al-lote-devuelto>")


if __name__ == "__main__":
    main()
