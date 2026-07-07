"""
evaluate_model.py
-----------------
Evaluación COMPLETA de un modelo entrenado sobre el TEST set:
  - Detección: Precision, Recall, mAP50, mAP50-95  (yolo val)
  - Conteo:    barrido de conf para hallar el óptimo, y en ese óptimo
               MAE, RMSE, MAPE, sesgo, error agregado del lote.

Acumula una fila por modelo en reports/comparacion_modelos.csv para poder
comparar arquitecturas de forma reproducible (material de tesis).

Uso:
  python scripts/evaluate_model.py --weights runs/detect/alevines_v8s/weights/best.pt --tag YOLOv8s
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--tag", required=True, help="Nombre del modelo para la tabla (ej. YOLOv8n)")
    p.add_argument("--data", default="data/dataset_yolo/data.yaml")
    p.add_argument("--test-images", default="data/dataset_yolo/images/test")
    p.add_argument("--manual-csv", default="data/counts/conteo_manual_test.csv")
    p.add_argument("--out-csv", default="reports/comparacion_modelos.csv")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default="0")
    p.add_argument("--confs", default="0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)

    # --- Detección en test ---
    r = model.val(data=args.data, split="test", imgsz=args.imgsz,
                  device=args.device, workers=0, verbose=False)
    det = {
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "mAP50": round(float(r.box.map50), 4),
        "mAP50_95": round(float(r.box.map), 4),
    }

    # --- Conteo: barrido de conf ---
    rows = list(csv.DictReader(open(args.manual_csv)))
    imgs = [(x["image_name"], int(x["real_count"])) for x in rows]
    tdir = Path(args.test_images)
    reals = np.array([c for _, c in imgs])

    best = None
    for conf in [float(c) for c in args.confs.split(",")]:
        preds = []
        for name, _ in imgs:
            im = cv2.imread(str(tdir / name))
            res = model.predict(im, conf=conf, iou=0.5, imgsz=args.imgsz,
                                device=args.device, verbose=False)[0]
            preds.append(int(res.boxes.shape[0]))
        preds = np.array(preds)
        err = preds - reals
        mae = float(np.abs(err).mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        mape = float((np.abs(err) / reals * 100).mean())
        bias = float(err.mean())
        # R² de la regresión conteo_predicho vs real (métrica estándar del dominio)
        ss_res = float(np.sum((preds - reals) ** 2))
        ss_tot = float(np.sum((reals - reals.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # Exactitud de conteo % = 100 - MAPE (media por imagen)
        acc = 100.0 - mape
        agg_err = abs(int(preds.sum()) - int(reals.sum()))
        agg_pct = agg_err / int(reals.sum()) * 100
        cand = dict(conf=conf, MAE=round(mae, 2), RMSE=round(rmse, 2),
                    MAPE=round(mape, 2), R2=round(r2, 4), acc_conteo=round(acc, 2),
                    bias=round(bias, 2),
                    pred_total=int(preds.sum()), real_total=int(reals.sum()),
                    agg_err=agg_err, agg_pct=round(agg_pct, 2))
        if best is None or cand["MAE"] < best["MAE"]:
            best = cand

    row = {"modelo": args.tag, **det, **best,
           "weights": args.weights}

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"\n===== {args.tag} =====")
    print(f"  DETECCIÓN  P={det['precision']} R={det['recall']} "
          f"mAP50={det['mAP50']} mAP50-95={det['mAP50_95']}")
    print(f"  CONTEO@conf={best['conf']}  MAE={best['MAE']} RMSE={best['RMSE']} "
          f"MAPE={best['MAPE']}% bias={best['bias']}")
    print(f"  Lote: real={best['real_total']} pred={best['pred_total']} "
          f"(error agregado {best['agg_err']} = {best['agg_pct']}%)")
    print(f"  -> fila agregada a {out}")


if __name__ == "__main__":
    main()
