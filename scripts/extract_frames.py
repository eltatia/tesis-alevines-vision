"""
extract_frames.py
-----------------
Extrae frames desde los videos en data/raw/videos/ y los guarda en data/frames/all/.

Opcional: deduplicación perceptual (pHash) para descartar frames casi idénticos
antes del etiquetado, reduciendo trabajo manual.

Uso desde la raíz del proyecto:
    python scripts/extract_frames.py
    python scripts/extract_frames.py --interval 15
    python scripts/extract_frames.py --interval 30 --dedup --phash-threshold 5

Argumentos:
    --interval         cada cuántos frames guardar uno (default: 30)
    --dedup            activa filtro de duplicados por pHash
    --phash-threshold  distancia Hamming mínima entre hashes (default: 5)
    --videos-dir       carpeta de videos (default: data/raw/videos)
    --output-dir       carpeta de salida (default: data/frames/all)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae frames desde videos.")
    parser.add_argument("--interval", type=int, default=30,
                        help="Guardar 1 frame cada N (default: 30)")
    parser.add_argument("--dedup", action="store_true",
                        help="Activa deduplicación perceptual con pHash")
    parser.add_argument("--phash-threshold", type=int, default=5,
                        help="Distancia Hamming mínima para considerar frames distintos (default: 5)")
    parser.add_argument("--videos-dir", type=str, default="data/raw/videos",
                        help="Carpeta con videos de entrada")
    parser.add_argument("--output-dir", type=str, default="data/frames/all",
                        help="Carpeta de salida para los frames")
    return parser.parse_args()


def extract_from_video(
    video_path: Path,
    output_dir: Path,
    interval: int,
    dedup: bool,
    phash_threshold: int,
) -> tuple[int, int]:
    """
    Devuelve (frames_guardados, frames_descartados_por_dedup).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir: {video_path.name}")
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = video_path.stem

    seen_hashes: list = []  # poblado solo si dedup=True
    if dedup:
        # Import perezoso: solo si el usuario activó --dedup
        from PIL import Image
        import imagehash

    frame_idx = 0
    saved = 0
    skipped_dedup = 0

    pbar = tqdm(total=total_frames, desc=f"  {video_name}", unit="f", leave=False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            keep = True

            if dedup:
                # BGR -> RGB para PIL
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                h = imagehash.phash(pil_img)
                # Comparar con todos los hashes ya guardados
                if any(abs(h - prev) < phash_threshold for prev in seen_hashes):
                    keep = False
                    skipped_dedup += 1
                else:
                    seen_hashes.append(h)

            if keep:
                out_path = output_dir / f"{video_name}_frame_{saved:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return saved, skipped_dedup


def main() -> None:
    args = parse_args()

    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not videos_dir.exists():
        print(f"[ERROR] No existe la carpeta de videos: {videos_dir.resolve()}")
        return

    videos = sorted([p for p in videos_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in VIDEO_EXTS])

    if not videos:
        print(f"[AVISO] No se encontraron videos en {videos_dir.resolve()}")
        print(f"        Extensiones soportadas: {sorted(VIDEO_EXTS)}")
        return

    print(f"Encontrados {len(videos)} videos en {videos_dir.resolve()}")
    print(f"Intervalo de frames: cada {args.interval}")
    print(f"Deduplicación pHash: {'ON (umbral=' + str(args.phash_threshold) + ')' if args.dedup else 'OFF'}")
    print(f"Salida: {output_dir.resolve()}\n")

    total_saved = 0
    total_skipped = 0

    for video_path in videos:
        saved, skipped = extract_from_video(
            video_path,
            output_dir,
            interval=args.interval,
            dedup=args.dedup,
            phash_threshold=args.phash_threshold,
        )
        print(f"  {video_path.name}: {saved} frames guardados"
              + (f" (descartados por dedup: {skipped})" if args.dedup else ""))
        total_saved += saved
        total_skipped += skipped

    print(f"\nTotal frames guardados: {total_saved}")
    if args.dedup:
        print(f"Total descartados por similitud: {total_skipped}")


if __name__ == "__main__":
    main()
