"""
predict_video.py
----------------
Detecta y cuenta alevines frame por frame en un video, guarda el video procesado
con cajas y un overlay de "Frame: N | Alevines: X | FPS: Y".

LIMITACIÓN CONOCIDA (v1):
    No usa tracking. Cada frame se cuenta de forma independiente, por lo que
    el mismo alevín se contará varias veces a lo largo del video. Documentar
    esto en la tesis. Para conteo total real sobre video, integrar luego
    ByteTrack/BoT-SORT (parámetro 'tracker' de Ultralytics).

Uso:
    python scripts/predict_video.py --weights runs/detect/alevines_run/weights/best.pt --source data/raw/videos/video01.mp4
    python scripts/predict_video.py --weights ... --source ... --conf 0.3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predicción y conteo en video.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path al .pt entrenado")
    parser.add_argument("--source", type=str, required=True,
                        help="Path al video de entrada")
    parser.add_argument("--output-dir", type=str, default="reports/imagenes_resultado",
                        help="Carpeta donde guardar el video procesado")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    source = Path(args.source)

    if not weights.exists():
        raise SystemExit(f"[ERROR] Pesos no encontrados: {weights}")
    if not source.exists() or source.suffix.lower() not in VIDEO_EXTS:
        raise SystemExit(f"[ERROR] Video no válido: {source}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{source.stem}_result.mp4"

    from ultralytics import YOLO
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] No se pudo abrir el video: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit("[ERROR] No se pudo abrir el writer de video. Verifica códecs.")

    print(f"Modelo: {weights}")
    print(f"Video : {source}  ({width}x{height} @ {src_fps:.1f} fps, {total_frames} frames)")
    print(f"Salida: {out_path}\n")

    frame_idx = 0
    counts: list[int] = []
    t_start = time.time()
    t_window_start = t_start
    fps_recent = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        r = results[0]
        boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        count = len(boxes_xyxy)
        counts.append(count)

        for (x1, y1, x2, y2) in boxes_xyxy:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          color=(0, 255, 0), thickness=2)

        # FPS de inferencia recalculado cada 30 frames
        if frame_idx > 0 and frame_idx % 30 == 0:
            t_now = time.time()
            fps_recent = 30.0 / (t_now - t_window_start)
            t_window_start = t_now

        overlay = f"Frame: {frame_idx + 1}/{total_frames}  Alevines: {count}  FPS: {fps_recent:.1f}"
        (tw, th), bl = cv2.getTextSize(overlay, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (10, 10), (10 + tw + 10, 10 + th + bl + 10), (0, 0, 0), -1)
        cv2.putText(frame, overlay, (15, 10 + th + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  procesados {frame_idx}/{total_frames} frames "
                  f"(último conteo: {count}, fps inferencia: {fps_recent:.1f})")

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    avg_count = sum(counts) / len(counts) if counts else 0.0
    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0

    print(f"\nFrames procesados: {frame_idx}")
    print(f"Tiempo total: {elapsed:.1f}s  (FPS promedio: {avg_fps:.2f})")
    print(f"Conteo por frame -> min: {min_count}, max: {max_count}, promedio: {avg_count:.1f}")
    print(f"Video resultado: {out_path}")
    print("\nNota: este conteo es por frame, NO es el conteo único de alevines en el video.")
    print("      Para conteo único integrar tracking (ByteTrack/BoT-SORT) en una v2.")


if __name__ == "__main__":
    main()
