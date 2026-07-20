"""
extract_new_videos.py
---------------------
Extrae frames DIVERSOS de videos específicos (por defecto los 2 nuevos),
saltando frames por seek (rápido en videos largos a alto fps) y aplicando
deduplicación perceptual (pHash) para quedarse solo con frames visualmente
distintos — así se etiqueta poco y variado.

Uso:
  python scripts/extract_new_videos.py
  python scripts/extract_new_videos.py --step 240 --max-per-video 60 --phash-threshold 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image
import imagehash


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="+",
                   default=["data/raw/videos/IMG_0177.MOV", "data/raw/videos/IMG_0178.MOV"])
    p.add_argument("--out", default="data/frames/nuevos")
    p.add_argument("--step", type=int, default=240,
                   help="Frames entre muestras candidatas (240 ≈ 2 s a 120 fps)")
    p.add_argument("--phash-threshold", type=int, default=8,
                   help="Distancia Hamming mínima para considerar frames distintos")
    p.add_argument("--max-per-video", type=int, default=60,
                   help="Tope de frames diversos a guardar por video")
    return p.parse_args()


def process(video: Path, out_dir: Path, step: int, thr: int, cap_max: int) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] No abre: {video.name}")
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    name = video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    seen: list = []
    saved = 0
    idx = 0
    while idx < total and saved < cap_max:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            idx += step
            continue
        # pHash sobre versión reducida
        rgb = cv2.cvtColor(cv2.resize(frame, (256, 456)), cv2.COLOR_BGR2RGB)
        h = imagehash.phash(Image.fromarray(rgb))
        if all(abs(h - prev) >= thr for prev in seen):
            seen.append(h)
            out = out_dir / f"{name}_frame_{saved:04d}.jpg"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        idx += step
    cap.release()
    print(f"  {video.name}: {saved} frames diversos guardados (de {total} totales)")
    return saved


def main():
    args = parse_args()
    out = Path(args.out)
    total = 0
    for v in args.videos:
        vp = Path(v)
        if not vp.exists():
            print(f"[AVISO] no existe: {v}")
            continue
        sub = out / vp.stem
        total += process(vp, sub, args.step, args.phash_threshold, args.max_per_video)
    print(f"\nTotal frames diversos: {total}")
    print(f"Salida: {out.resolve()}")


if __name__ == "__main__":
    main()
