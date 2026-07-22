"""
compare_trackers.py
-------------------
Compara trackers (ByteTrack vs BoT-SORT) para el CONTEO por video: en vez de
contar por frame (que duplica el mismo pez en frames consecutivos), se cuentan
IDs UNICOS a lo largo del video.

Metrica clave = FRAGMENTACION. En una bandeja estatica el nº real de peces es
~constante, asi que:
  - detecciones_por_frame (mediana) ≈ nº real de peces visibles.
  - IDs_unicos deberia ≈ ese numero.
  - frag_ratio = IDs_unicos / mediana_por_frame.
      1.0  = tracking perfecto (IDs estables)
      >1   = el tracker fragmenta (pierde peces y les da ID nuevo -> SOBRECUENTA)

Uso:
  python scripts/compare_trackers.py --weights best.pt --video data/raw/videos/IMG_0177.MOV \
      --trackers bytetrack.yaml botsort.yaml --max-frames 400 --vid-stride 4 --imgsz 1280
"""
from __future__ import annotations
import argparse, time, csv
from pathlib import Path
import numpy as np


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--video',required=True)
    p.add_argument('--trackers',nargs='+',default=['bytetrack.yaml','botsort.yaml'])
    p.add_argument('--max-frames',type=int,default=400)
    p.add_argument('--vid-stride',type=int,default=4)
    p.add_argument('--conf',type=float,default=0.3)
    p.add_argument('--iou',type=float,default=0.5)
    p.add_argument('--imgsz',type=int,default=1280)
    p.add_argument('--device',default='0')
    p.add_argument('--out',default='reports/trackers_comparacion.csv')
    return p.parse_args()


def run_tracker(weights,video,tracker,args):
    from ultralytics import YOLO
    model=YOLO(weights)
    unique=set(); per_frame=[]; nframes=0
    t0=time.time()
    gen=model.track(source=video,tracker=tracker,persist=True,conf=args.conf,
                    iou=args.iou,imgsz=args.imgsz,device=args.device,
                    stream=True,vid_stride=args.vid_stride,verbose=False)
    for r in gen:
        nframes+=1
        b=r.boxes
        per_frame.append(0 if b is None else len(b))
        if b is not None and b.id is not None:
            for tid in b.id.int().tolist(): unique.add(tid)
        if nframes>=args.max_frames: break
    dt=time.time()-t0
    pf=np.array(per_frame) if per_frame else np.array([0])
    med=float(np.median(pf))
    return {
        'tracker':tracker,
        'frames':nframes,
        'IDs_unicos':len(unique),
        'det_mediana_frame':round(med,1),
        'det_media_frame':round(float(pf.mean()),1),
        'det_max_frame':int(pf.max()),
        'frag_ratio':round(len(unique)/med,2) if med>0 else None,
        'fps':round(nframes/dt,1) if dt>0 else None,
    }


def main():
    args=parse_args()
    print(f'Video: {args.video}')
    print(f'Modelo: {args.weights} | frames={args.max_frames} stride={args.vid_stride} imgsz={args.imgsz}\n')
    rows=[]
    for tr in args.trackers:
        print(f'>> corriendo {tr} ...')
        r=run_tracker(args.weights,args.video,tr,args)
        r['video']=Path(args.video).stem
        rows.append(r)
        print(f'   IDs_unicos={r["IDs_unicos"]} | det_mediana/frame={r["det_mediana_frame"]} '
              f'| frag_ratio={r["frag_ratio"]} | fps={r["fps"]}')
    # guardar
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    write_header=not out.exists()
    with out.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['video','tracker','frames','IDs_unicos',
            'det_mediana_frame','det_media_frame','det_max_frame','frag_ratio','fps'])
        if write_header: w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\n=== RESUMEN ({Path(args.video).stem}) ===')
    best=min(rows,key=lambda r:abs((r['frag_ratio'] or 99)-1.0))
    for r in rows:
        mark=' <- mejor (IDs mas estables)' if r is best else ''
        print(f"  {r['tracker']:<16} IDs_unicos={r['IDs_unicos']:<4} frag_ratio={r['frag_ratio']} fps={r['fps']}{mark}")
    print(f'-> guardado en {out}')


if __name__=='__main__':
    main()
