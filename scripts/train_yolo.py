"""
train_yolo.py
-------------
Entrena un modelo YOLO de Ultralytics con el dataset de alevines.

Uso:
    python scripts/train_yolo.py
    python scripts/train_yolo.py --model yolov8n.pt --epochs 100 --imgsz 640 --batch 8
    python scripts/train_yolo.py --model yolov8s.pt --epochs 150 --imgsz 800 --batch 4
    python scripts/train_yolo.py --model yolo11n.pt --epochs 100        # YOLOv11

Argumentos clave:
    --model    pesos base preentrenados (yolov8n.pt, yolov8s.pt, yolo11n.pt, ...)
    --data     path a data.yaml (default: data/dataset_yolo/data.yaml)
    --epochs   épocas de entrenamiento
    --imgsz    tamaño de imagen
    --batch    tamaño de batch (reducir si la GPU/CPU se queda sin memoria)
    --device   "cpu", "0" para GPU 0, "0,1" para multi-GPU
    --name     nombre del experimento dentro de runs/detect/

El resultado queda en runs/detect/<name>/, con best.pt y last.pt en weights/.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena YOLO para detección de alevines.")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Pesos base (yolov8n.pt | yolov8s.pt | yolo11n.pt | ...)")
    parser.add_argument("--data", type=str, default="data/dataset_yolo/data.yaml",
                        help="Path al data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Épocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cpu' o índice de GPU ('0', '0,1')")
    parser.add_argument("--name", type=str, default="alevines_run",
                        help="Nombre del experimento bajo runs/detect/")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping: épocas sin mejora antes de parar")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad")
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar desde la última corrida con el mismo --name")
    parser.add_argument("--workers", type=int, default=0,
                        help="Workers del DataLoader. En Windows usa 0 para evitar crashes "
                             "de multiprocessing (default: 0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"[ERROR] No se encuentra data.yaml en: {data_path.resolve()}")

    # Import perezoso: ultralytics tarda en cargar
    from ultralytics import YOLO

    print(f"Modelo base: {args.model}")
    print(f"Dataset:     {data_path.resolve()}")
    print(f"Épocas:      {args.epochs} | imgsz: {args.imgsz} | batch: {args.batch}")
    print(f"Device:      {args.device}")
    print(f"Experimento: runs/detect/{args.name}\n")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        resume=args.resume,
        workers=args.workers,
        # Augmentations YOLO por defecto (mosaic, mixup, hsv, flip) ya activos.
        # Aquí puedes sobrescribir si hace falta:
        # mosaic=1.0, mixup=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        # fliplr=0.5, flipud=0.0,
        verbose=True,
    )

    print("\nEntrenamiento finalizado.")
    print(f"Pesos finales en: runs/detect/{args.name}/weights/best.pt")
    print("Sugerencia: ejecuta evaluación con:")
    print(f"  yolo detect val model=runs/detect/{args.name}/weights/best.pt data={args.data}")


if __name__ == "__main__":
    main()
