"""
auto_label_fastsam.py
---------------------
Auto-etiquetado con FastSAM (Ultralytics). A diferencia de SAM 2 vía
`model(img)`, FastSAM SI tiene modo "segment everything" nativo:
genera todas las máscaras de la imagen sin necesidad de prompts.

Estrategia:
    1. FastSAM segmenta todo lo que ve.
    2. Filtros por:
       - área (descarta ruido y bandeja)
       - aspect ratio (alevines son alargados)
       - posición (descarta máscaras que tocan el borde si son demasiado grandes)
    3. Convierte máscaras → bounding boxes YOLO normalizadas.

Uso:
    # Test rápido sobre 5 imágenes con visualizaciones
    python scripts/auto_label_fastsam.py --src data/labeling_test --visualize --overwrite

    # Procesar todo
    python scripts/auto_label_fastsam.py --src data/labeling

    # Ajustar imgsz si los alevines son muy pequeños relativos a la imagen
    python scripts/auto_label_fastsam.py --src data/labeling --imgsz 1536

Modelos disponibles (Ultralytics los descarga automáticamente):
    FastSAM-s.pt  (~23 MB, rápido)
    FastSAM-x.pt  (~138 MB, más preciso) ← default

Parámetros de FastSAM:
    --imgsz       tamaño de inferencia (default: 1024). Para imágenes grandes
                  con objetos pequeños, subir a 1536 o 2048.
    --conf        confianza mínima de máscara (default: 0.4)
    --iou         IoU para NMS (default: 0.9)
    --retina-masks   máscaras de mayor resolución (default: True)

Filtros aplicados después de FastSAM:
    --min-area-pct  área mínima como fracción del área total (default: 0.00005)
    --max-area-pct  área máxima (default: 0.02)
    --min-aspect    aspect ratio mínimo largo/ancho del bbox (default: 1.4)
    --max-aspect    aspect ratio máximo (default: 6.0)

Salida:
    <src>/<imagen>.txt        etiquetas YOLO (class_id=0)
    reports/auto_label_viz/   visualizaciones si --visualize
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-etiquetado con FastSAM.")
    p.add_argument("--src", type=str, default="data/labeling")
    p.add_argument("--model", type=str, default="FastSAM-x.pt",
                   help="FastSAM-s.pt | FastSAM-x.pt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--viz-dir", type=str, default="reports/auto_label_viz")
    # Parámetros de inferencia FastSAM
    p.add_argument("--imgsz", type=int, default=1024,
                   help="Tamaño de inferencia (subir a 1536/2048 si los alevines son pequeños)")
    p.add_argument("--conf", type=float, default=0.4,
                   help="Confianza mínima de máscara (default: 0.4)")
    p.add_argument("--iou", type=float, default=0.9,
                   help="IoU para NMS (default: 0.9)")
    p.add_argument("--no-retina", action="store_true",
                   help="Desactivar retina_masks (más rápido pero menos preciso)")
    # Filtros post-procesamiento
    p.add_argument("--min-area-pct", type=float, default=0.00005)
    p.add_argument("--max-area-pct", type=float, default=0.02)
    p.add_argument("--min-aspect", type=float, default=1.4)
    p.add_argument("--max-aspect", type=float, default=6.0)
    p.add_argument("--touch-edge-frac", type=float, default=0.5)
    p.add_argument("--nms-iou", type=float, default=0.4,
                   help="IoU para NMS post-filtros (default: 0.4). Cajas que se solapan más se fusionan.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def nms_boxes(bboxes: list[tuple[int, int, int, int]], iou_threshold: float) -> list[tuple[int, int, int, int]]:
    """NMS clásico: ordena por área descendente, elimina solapamientos altos."""
    if not bboxes:
        return []
    boxes = np.array(bboxes, dtype=np.float32)
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    order = areas.argsort()[::-1]  # mayor área primero (más confiable)

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w_int = np.maximum(0.0, xx2 - xx1 + 1)
        h_int = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w_int * h_int
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)

        order = order[1:][iou < iou_threshold]

    return [tuple(int(v) for v in boxes[i]) for i in keep]


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def touches_edge(bbox: tuple[int, int, int, int], w: int, h: int) -> bool:
    x_min, y_min, x_max, y_max = bbox
    return x_min == 0 or y_min == 0 or x_max == w - 1 or y_max == h - 1


def filter_masks(
    masks: np.ndarray,
    img_w: int,
    img_h: int,
    args: argparse.Namespace,
) -> list[tuple[int, int, int, int]]:
    total_area = img_w * img_h
    accepted: list[tuple[int, int, int, int]] = []

    for i in range(masks.shape[0]):
        mask = masks[i].astype(bool)
        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        bw = x_max - x_min + 1
        bh = y_max - y_min + 1
        mask_area = int(mask.sum())
        area_frac = mask_area / total_area

        if area_frac < args.min_area_pct or area_frac > args.max_area_pct:
            continue

        if touches_edge(bbox, img_w, img_h) and area_frac > args.touch_edge_frac:
            continue

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
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"[ERROR] No hay imágenes en {src}")

    viz_dir = Path(args.viz_dir)
    if args.visualize:
        viz_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando FastSAM...")
    from ultralytics import FastSAM
    model = FastSAM(args.model)
    print(f"Modelo: {args.model} | device: {args.device} | imgsz: {args.imgsz}")
    print(f"FastSAM: conf={args.conf}, iou={args.iou}, retina={not args.no_retina}")
    print(f"Filtros: area=[{args.min_area_pct:.5%}, {args.max_area_pct:.2%}]  "
          f"aspect=[{args.min_aspect}, {args.max_aspect}]\n")
    print(f"Procesando {len(images)} imágenes")

    counts: list[int] = []
    stats = {"processed": 0, "total_detections": 0, "with_zero": 0}

    for img_path in tqdm(images, desc="FastSAM", unit="img"):
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists() and not args.overwrite:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[AVISO] No se pudo leer: {img_path.name}")
            continue
        h, w = img.shape[:2]

        # FastSAM en modo "everything" — sin prompts, devuelve todas las máscaras
        results = model(
            img,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            retina_masks=not args.no_retina,
            verbose=False,
        )
        r = results[0]

        if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
            txt_path.write_text("")
            stats["processed"] += 1
            stats["with_zero"] += 1
            counts.append(0)
            continue

        masks_np = r.masks.data.cpu().numpy()  # shape (N, mh, mw), float
        mh, mw = masks_np.shape[1], masks_np.shape[2]
        scale_x = w / mw
        scale_y = h / mh

        # Trabajar con bboxes en la resolución de la máscara para no explotar la memoria.
        # Solo se escala al final el bbox (4 floats) — no las máscaras completas.
        bboxes_mask: list[tuple[int, int, int, int, int]] = []  # x_min, y_min, x_max, y_max, area_mask_px
        for i in range(masks_np.shape[0]):
            m = masks_np[i] > 0.5
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue
            x_min, y_min = int(xs.min()), int(ys.min())
            x_max, y_max = int(xs.max()), int(ys.max())
            area = int(m.sum())
            bboxes_mask.append((x_min, y_min, x_max, y_max, area))

        # Escalar a coordenadas de la imagen original y filtrar
        total_area_orig = w * h
        total_area_mask = mh * mw
        bboxes: list[tuple[int, int, int, int]] = []
        for (xm0, ym0, xm1, ym1, area_mask) in bboxes_mask:
            # Bbox escalado a original
            x_min = int(xm0 * scale_x)
            y_min = int(ym0 * scale_y)
            x_max = int(xm1 * scale_x)
            y_max = int(ym1 * scale_y)
            bw = x_max - x_min + 1
            bh = y_max - y_min + 1

            # Estimar fracción de área usando la máscara en su resolución original
            area_frac = area_mask / total_area_mask

            if area_frac < args.min_area_pct or area_frac > args.max_area_pct:
                continue

            # Borde + área grande -> bandeja
            on_edge = (xm0 == 0 or ym0 == 0 or xm1 == mw - 1 or ym1 == mh - 1)
            if on_edge and area_frac > args.touch_edge_frac:
                continue

            longer = max(bw, bh)
            shorter = min(bw, bh)
            if shorter == 0:
                continue
            aspect = longer / shorter
            if aspect < args.min_aspect or aspect > args.max_aspect:
                continue

            bboxes.append((x_min, y_min, x_max, y_max))

        # NMS para fusionar cajas duplicadas sobre el mismo alevín
        bboxes = nms_boxes(bboxes, iou_threshold=args.nms_iou)

        lines = [f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                 for (xc, yc, bw, bh) in (xyxy_to_yolo(b, w, h) for b in bboxes)]
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        if args.visualize:
            viz = draw_viz(img, bboxes)
            cv2.imwrite(str(viz_dir / f"{img_path.stem}_viz.jpg"), viz,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

        stats["processed"] += 1
        stats["total_detections"] += len(bboxes)
        if len(bboxes) == 0:
            stats["with_zero"] += 1
        counts.append(len(bboxes))

    print(f"\n=== Resumen ===")
    print(f"Imágenes procesadas:        {stats['processed']}")
    print(f"Detecciones totales:        {stats['total_detections']}")
    print(f"Imágenes con 0 detecciones: {stats['with_zero']}")
    if counts:
        print(f"Detecciones por imagen -> "
              f"min: {min(counts)}, max: {max(counts)}, promedio: {sum(counts)/len(counts):.1f}")
    if args.visualize:
        print(f"\nVisualizaciones en: {viz_dir.resolve()}")


if __name__ == "__main__":
    main()
