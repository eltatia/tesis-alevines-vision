"""
predict_image.py
----------------
Detecta y cuenta alevines en una imagen (o todas las imágenes de una carpeta)
usando un modelo YOLO entrenado.

Uso:
    python scripts/predict_image.py --weights runs/detect/alevines_run/weights/best.pt --source data/raw/images/img_001.jpg
    python scripts/predict_image.py --weights ... --source data/dataset_yolo/images/test --conf 0.3

Salida:
    - Imágenes con cajas dibujadas y texto "Alevines detectados: N"
      en reports/imagenes_resultado/<nombre>_result.jpg
    - Mensaje en consola con el conteo por imagen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predicción y conteo en imagen(es).")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path al .pt entrenado (e.g. runs/detect/alevines_run/weights/best.pt)")
    parser.add_argument("--source", type=str, required=True,
                        help="Imagen única o carpeta con imágenes")
    parser.add_argument("--output-dir", type=str, default="reports/imagenes_resultado",
                        help="Carpeta donde guardar los resultados")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Umbral de confianza (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="Umbral IoU para NMS (default: 0.5)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tamaño de imagen para inferencia")
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cpu' o índice de GPU")
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTS:
            return [source]
        raise SystemExit(f"[ERROR] El archivo no es una imagen soportada: {source}")
    if source.is_dir():
        return sorted([p for p in source.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    raise SystemExit(f"[ERROR] No existe el origen: {source}")


def draw_result(img, boxes_xyxy, count: int):
    """Dibuja cajas y el texto del conteo sobre la imagen (in-place safe — devuelve copia)."""
    out = img.copy()
    for (x1, y1, x2, y2) in boxes_xyxy:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      color=(0, 255, 0), thickness=2)

    label = f"Alevines detectados: {count}"
    # Fondo del texto para legibilidad
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(out, (10, 10), (10 + tw + 10, 10 + th + baseline + 10),
                  color=(0, 0, 0), thickness=-1)
    cv2.putText(out, label, (15, 10 + th + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"[ERROR] Pesos no encontrados: {weights}")

    source = Path(args.source)
    images = collect_images(source)
    if not images:
        raise SystemExit(f"[ERROR] No hay imágenes en: {source}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(weights))

    print(f"Modelo: {weights}")
    print(f"Imágenes a procesar: {len(images)}")
    print(f"conf={args.conf}  iou={args.iou}  imgsz={args.imgsz}  device={args.device}\n")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[AVISO] No se pudo leer: {img_path.name}")
            continue

        results = model.predict(
            source=img,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        r = results[0]
        boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        count = len(boxes_xyxy)

        result_img = draw_result(img, boxes_xyxy, count)
        out_path = output_dir / f"{img_path.stem}_result.jpg"
        cv2.imwrite(str(out_path), result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"  {img_path.name}: {count} alevines -> {out_path.relative_to(output_dir.parent.parent) if output_dir.is_absolute() is False else out_path}")

    print(f"\nResultados guardados en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
