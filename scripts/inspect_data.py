"""
inspect_data.py
---------------
Audita el contenido de data/raw/{images,videos}/ y produce un reporte detallado:

- Conteo y peso total por tipo.
- Distribución de resoluciones y orientaciones de imágenes.
- Distribución de tamaño en MB.
- Detección de ráfagas (imágenes tomadas en el mismo segundo) → candidatas a deduplicación.
- Metadatos de video: duración, fps, resolución, codec, frames totales.
- Estimación de cuántos frames extraería extract_frames.py con distintos intervalos.

Uso:
    python scripts/inspect_data.py
    python scripts/inspect_data.py --images-dir data/raw/images --videos-dir data/raw/videos
    python scripts/inspect_data.py --json reports/inventario.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}

# Regex para timestamps tipo 20260505_154136 (Samsung/Android nativo)
TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audita las imágenes y videos crudos.")
    p.add_argument("--images-dir", type=str, default="data/raw/images")
    p.add_argument("--videos-dir", type=str, default="data/raw/videos")
    p.add_argument("--json", type=str, default=None,
                   help="Si se indica, guarda el reporte también como JSON")
    return p.parse_args()


def fourcc_to_str(fourcc_int: int) -> str:
    try:
        return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
    except Exception:
        return str(fourcc_int)


def format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def inspect_images(images_dir: Path) -> dict:
    files = sorted([p for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

    if not files:
        return {"count": 0, "files": []}

    resolutions: Counter[str] = Counter()
    orientations: Counter[str] = Counter()
    size_buckets: Counter[str] = Counter()
    timestamps_by_second: dict[str, list[str]] = defaultdict(list)
    total_bytes = 0
    errors: list[str] = []

    for f in files:
        total_bytes += f.stat().st_size
        try:
            with Image.open(f) as im:
                # Aplicar rotación EXIF para reportar dimensiones "visuales"
                # (lo que vé el usuario y lo que Ultralytics usa al entrenar).
                im = ImageOps.exif_transpose(im)
                w, h = im.size
        except Exception as e:
            errors.append(f"{f.name}: {e}")
            continue

        resolutions[f"{w}x{h}"] += 1
        orientations["vertical (portrait)" if h > w else
                     "horizontal (landscape)" if w > h else "cuadrada"] += 1

        size_mb = f.stat().st_size / (1024 * 1024)
        if size_mb < 1:
            bucket = "<1 MB"
        elif size_mb < 3:
            bucket = "1-3 MB"
        elif size_mb < 5:
            bucket = "3-5 MB"
        else:
            bucket = "5+ MB"
        size_buckets[bucket] += 1

        m = TIMESTAMP_RE.search(f.name)
        if m:
            timestamps_by_second[m.group(0)].append(f.name)

    bursts = {ts: files_ for ts, files_ in timestamps_by_second.items() if len(files_) > 1}

    return {
        "count": len(files),
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "avg_mb": round((total_bytes / (1024 * 1024)) / len(files), 2),
        "resolutions": dict(resolutions),
        "orientations": dict(orientations),
        "size_buckets": dict(size_buckets),
        "unique_timestamps": len(timestamps_by_second),
        "bursts_count": len(bursts),
        "bursts_total_extra_files": sum(len(v) - 1 for v in bursts.values()),
        "bursts_sample": dict(list(bursts.items())[:5]),
        "errors": errors,
    }


def inspect_videos(videos_dir: Path) -> dict:
    files = sorted([p for p in videos_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS])

    if not files:
        return {"count": 0, "files": []}

    videos_info = []
    total_bytes = 0
    total_duration = 0.0
    total_frames = 0

    for f in files:
        total_bytes += f.stat().st_size
        cap = cv2.VideoCapture(str(f))
        if not cap.isOpened():
            videos_info.append({"name": f.name, "error": "no se pudo abrir"})
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        duration = (frames / fps) if fps > 0 else 0.0

        cap.release()

        total_duration += duration
        total_frames += frames

        videos_info.append({
            "name": f.name,
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            "resolution": f"{width}x{height}",
            "orientation": "vertical" if height > width else "horizontal" if width > height else "cuadrada",
            "fps": round(fps, 2),
            "frames": frames,
            "duration": format_duration(duration),
            "duration_seconds": round(duration, 2),
            "codec": fourcc_to_str(fourcc),
            "frames_with_interval_15": frames // 15 if frames else 0,
            "frames_with_interval_30": frames // 30 if frames else 0,
            "frames_with_interval_60": frames // 60 if frames else 0,
        })

    return {
        "count": len(files),
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "total_duration": format_duration(total_duration),
        "total_duration_seconds": round(total_duration, 2),
        "total_frames": total_frames,
        "videos": videos_info,
    }


def print_report(img_report: dict, vid_report: dict) -> None:
    print("=" * 70)
    print("  AUDITORÍA DE DATOS - data/raw/")
    print("=" * 70)

    print(f"\n[IMÁGENES]  {img_report.get('count', 0)} archivos  ({img_report.get('total_mb', 0)} MB total)")
    if img_report.get("count", 0) > 0:
        print(f"  Tamaño promedio por imagen: {img_report['avg_mb']} MB")
        print(f"  Orientaciones: {img_report['orientations']}")
        print(f"\n  Resoluciones únicas ({len(img_report['resolutions'])}):")
        for res, n in sorted(img_report['resolutions'].items(), key=lambda x: -x[1]):
            print(f"    {res}: {n} imágenes")
        print(f"\n  Distribución de tamaño:")
        for bucket, n in img_report['size_buckets'].items():
            print(f"    {bucket}: {n}")
        print(f"\n  Timestamps únicos: {img_report['unique_timestamps']}")
        print(f"  Ráfagas detectadas (mismo segundo): {img_report['bursts_count']}")
        print(f"  Archivos extra en ráfagas: {img_report['bursts_total_extra_files']}")
        print(f"  -> estimacion de imagenes unicas si se descartan rafagas: "
              f"{img_report['count'] - img_report['bursts_total_extra_files']}")
        if img_report['bursts_sample']:
            print(f"\n  Ejemplos de ráfagas (primeros 5):")
            for ts, files in img_report['bursts_sample'].items():
                print(f"    {ts}: {len(files)} imágenes")
        if img_report['errors']:
            print(f"\n  [ERRORES] {len(img_report['errors'])} imágenes no se pudieron leer:")
            for e in img_report['errors'][:10]:
                print(f"    {e}")

    print(f"\n[VIDEOS]  {vid_report.get('count', 0)} archivos  ({vid_report.get('total_mb', 0)} MB total)")
    if vid_report.get("count", 0) > 0:
        print(f"  Duración total: {vid_report['total_duration']} ({vid_report['total_duration_seconds']}s)")
        print(f"  Frames totales en todos los videos: {vid_report['total_frames']:,}")
        for v in vid_report['videos']:
            if "error" in v:
                print(f"\n  {v['name']}: ERROR - {v['error']}")
                continue
            print(f"\n  {v['name']}  ({v['size_mb']} MB)")
            print(f"    Resolución: {v['resolution']} ({v['orientation']})")
            print(f"    Duración:   {v['duration']}   |  FPS: {v['fps']}   |  Frames: {v['frames']:,}")
            print(f"    Codec:      {v['codec']}")
            print(f"    Frames extraibles -> interval 15: {v['frames_with_interval_15']:,}"
                  f"  |  30: {v['frames_with_interval_30']:,}"
                  f"  |  60: {v['frames_with_interval_60']:,}")

    print("\n" + "=" * 70)
    print("  RESUMEN PARA DECISIONES")
    print("=" * 70)

    img_unique = (img_report.get('count', 0) -
                  img_report.get('bursts_total_extra_files', 0))
    vid_frames_30 = sum(v.get('frames_with_interval_30', 0)
                        for v in vid_report.get('videos', []))

    print(f"\n  Imágenes raw únicas (post-dedup estimado): ~{img_unique}")
    print(f"  Frames extraíbles de videos con interval=30: ~{vid_frames_30}")
    print(f"  Total potencial para dataset (antes de curación manual): ~{img_unique + vid_frames_30}")
    print()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    videos_dir = Path(args.videos_dir)

    if not images_dir.exists() and not videos_dir.exists():
        raise SystemExit(f"[ERROR] No existen ni {images_dir} ni {videos_dir}")

    img_report = inspect_images(images_dir) if images_dir.exists() else {"count": 0}
    vid_report = inspect_videos(videos_dir) if videos_dir.exists() else {"count": 0}

    print_report(img_report, vid_report)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"images": img_report, "videos": vid_report}, f,
                      indent=2, ensure_ascii=False)
        print(f"  Reporte JSON guardado en: {out}\n")


if __name__ == "__main__":
    main()
