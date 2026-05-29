"""
auto_label_sam2.py
------------------
Auto-etiqueta imágenes de alevines usando SAM 2 (Segment Anything Model 2)
de Meta, vía Ultralytics. Genera archivos .txt en formato YOLO con bounding
boxes derivadas de las máscaras.

Estrategia:
    1. SAM 2 segmenta TODO lo que ve en la imagen (modo "everything").
    2. Filtramos las máscaras por:
       - Área (descartar muy chicas: ruido; muy grandes: la bandeja)
       - Aspect ratio (alevines son alargados, ~1.5-5:1)
       - Posición (descartar máscaras que tocan el borde si son la bandeja)
    3. Convertimos cada máscara aceptada a bounding box YOLO normalizado.
    4. Guardamos un .txt por imagen junto a ella.

IMPORTANTE: Las etiquetas generadas son PROVISIONALES. Hay que revisarlas
en LabelImg o similar antes de entrenar.

Uso:
    # Test sobre 5 imágenes y guardar visualizaciones para revisar
    python scripts/auto_label_sam2.py --src data/frames/selected --limit 5 --visualize

    # Procesar todo el conjunto
    python scripts/auto_label_sam2.py --src data/frames/selected

    # Ajustar filtros si hay muchos falsos positivos
    python scripts/auto_label_sam2.py --src data/frames/selected --min-area-pct 0.0001 --max-area-pct 0.05 --min-aspect 1.3 --max-aspect 6.0

Parámetros clave de filtrado (porcentaje del área total de imagen):
    --min-area-pct    descarta máscaras muy chicas (ruido)
    --max-area-pct    descarta máscaras muy grandes (bandeja, fondo)
    --min-aspect      ratio largo/ancho mínimo (alevines son alargados)
    --max-aspect      ratio largo/ancho máximo

Salida:
    <src>/<imagen>.txt        etiquetas YOLO (class_id=0)
    reports/auto_label_viz/   visualizaciones para revisar (si --visualize)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-etiquetado con SAM 2.")
    p.add_argument("--src", type=str, default="data/frames/selected",
                   help="Carpeta con imágenes a etiquetar")
    p.add_argument("--model", type=str, default="sam2_b.pt",
                   help="Modelo SAM 2 (sam2_t.pt | sam2_s.pt | sam2_b.pt | sam2_l.pt)")
    p.add_argument("--device", type=str, default="cuda",
                   help="'cuda' o 'cpu' (default: cuda)")
    p.add_argument("--limit", type=int, default=0,
                   help="Procesar solo las primeras N imágenes (0 = todas)")
    p.add_argument("--visualize", action="store_true",
                   help="Guarda imágenes con cajas dibujadas para revisión visual")
    p.add_argument("--viz-dir", type=str, default="reports/auto_label_viz",
                   help="Carpeta de visualizaciones")
    # Filtros — ajustar según experimentación
    p.add_argument("--min-area-pct", type=float, default=0.00005,
                   help="Área mínima de la máscara como fracción del área total (default: 0.00005 = 0.005%%)")
    p.add_argument("--max-area-pct", type=float, default=0.02,
                   help="Área máxima de la máscara (default: 0.02 = 2%%)")
    p.add_argument("--min-aspect", type=float, default=1.4,
                   help="Aspect ratio mínimo largo/ancho del bbox (default: 1.4)")
    p.add_argument("--max-aspect", type=float, default=6.0,
                   help="Aspect ratio máximo (default: 6.0)")
    p.add_argument("--touch-edge-frac", type=float, default=0.5,
                   help="Si una máscara toca el borde de la imagen Y su área supera "
                        "esta fracción del área total, se descarta (probablemente bandeja). Default: 0.5")
    p.add_argument("--overwrite", action="store_true",
                   help="Sobrescribir .txt existentes")
    return p.parse_args()


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Devuelve (x_min, y_min, x_max, y_max) de la máscara binaria, o None si está vacía."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def touches_edge(bbox: tuple[int, int, int, int], w: int, h: int) -> bool:
    x_min, y_min, x_max, y_max = bbox
    return x_min == 0 or y_min == 0 or x_max == w - 1 or y_max == h - 1


def filter_masks(
    masks: list[np.ndarray],
    img_w: int,
    img_h: int,
    args: argparse.Namespace,
) -> list[tuple[int, int, int, int]]:
    """Aplica los filtros y devuelve lista de bboxes (xyxy) aceptadas."""
    total_area = img_w * img_h
    accepted: list[tuple[int, int, int, int]] = []

    for mask in masks:
        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        bw = x_max - x_min + 1
        bh = y_max - y_min + 1
        mask_area = int(mask.sum())
        area_frac = mask_area / total_area

        # Filtro de tamaño
        if area_frac < args.min_area_pct or area_frac > args.max_area_pct:
            continue

        # Filtro borde + área grande (bandeja)
        if touches_edge(bbox, img_w, img_h) and area_frac > args.touch_edge_frac:
            continue

        # Filtro de aspect ratio (largo/ancho del bbox)
        longer = max(bw, bh)
        shorter = min(bw, bh)
        if shorter == 0:
            continue
        aspect = longer / shorter
        if aspect < args.min_aspect or aspect > args.max_aspect:
            continue

        accepted.append(bbox)

    return accepted


def xyxy_to_yolo(bbox: tuple[int, int, int, int], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox
    x_c = ((x_min + x_max) / 2.0) / img_w
    y_c = ((y_min + y_max) / 2.0) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return x_c, y_c, w, h


def draw_viz(img: np.ndarray, bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"Detectados: {len(bboxes)}"
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(out, (10, 10), (10 + tw + 10, 10 + th + bl + 10), (0, 0, 0), -1)
    cv2.putText(out, label, (15, 10 + th + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"[ERROR] No existe: {src}")

    images = sorted([p for p in src.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise SystemExit(f"[ERROR] No hay imágenes en {src}")

    if args.limit > 0:
        images = images[:args.limit]

    viz_dir = Path(args.viz_dir)
    if args.visualize:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Import perezoso de Ultralytics (puede tardar al cargar SAM)
    print("Cargando modelo SAM 2...")
    from ultralytics import SAM

    model = SAM(args.model)
    # Si la primera vez no está el peso, Ultralytics lo descarga automáticamente.
    print(f"Modelo cargado: {args.model}\nDevice: {args.device}\n")

    print(f"Procesando {len(images)} imágenes")
    print(f"Filtros: area=[{args.min_area_pct:.5%}, {args.max_area_pct:.2%}]  "
          f"aspect=[{args.min_aspect}, {args.max_aspect}]\n")

    stats: dict[str, int] = {"detections_total": 0, "images_processed": 0, "images_with_zero": 0}
    counts: list[int] = []

    for img_path in tqdm(images, desc="SAM 2", unit="img"):
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists() and not args.overwrite:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[AVISO] No se pudo leer: {img_path.name}")
            continue
        h, w = img.shape[:2]

        # SAM 2 en modo "everything": sin prompts, segmenta todo lo que ve.
        results = model(img, device=args.device, verbose=False)
        r = results[0]

        if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
            txt_path.write_text("")
            stats["images_processed"] += 1
            stats["images_with_zero"] += 1
            counts.append(0)
            continue

        # Las máscaras vienen en escala del modelo; el atributo .data ya está en (N, H, W)
        masks_np = r.masks.data.cpu().numpy().astype(bool)

        bboxes = filter_masks(list(masks_np), w, h, args)

        # Escribir YOLO .txt
        lines = []
        for bbox in bboxes:
            xc, yc, bw, bh = xyxy_to_yolo(bbox, w, h)
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        if args.visualize:
            viz = draw_viz(img, bboxes)
            cv2.imwrite(str(viz_dir / f"{img_path.stem}_viz.jpg"), viz,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

        stats["detections_total"] += len(bboxes)
        stats["images_processed"] += 1
        if len(bboxes) == 0:
            stats["images_with_zero"] += 1
        counts.append(len(bboxes))

    print("\n=== Resumen ===")
    print(f"Imágenes procesadas:        {stats['images_processed']}")
    print(f"Detecciones totales:        {stats['detections_total']}")
    print(f"Imágenes con 0 detecciones: {stats['images_with_zero']}")
    if counts:
        print(f"Detecciones por imagen -> "
              f"min: {min(counts)}, max: {max(counts)}, promedio: {sum(counts)/len(counts):.1f}")
    if args.visualize:
        print(f"\nVisualizaciones en: {viz_dir.resolve()}")
        print("Revisa varias para evaluar calidad. Si hay muchos falsos positivos/negativos,")
        print("ajusta --min-area-pct, --max-area-pct, --min-aspect, --max-aspect.")


if __name__ == "__main__":
    main()
