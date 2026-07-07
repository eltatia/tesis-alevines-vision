"""
predict_sahi_counting.py
------------------------
Conteo con SAHI (Slicing Aided Hyper Inference): parte cada imagen en tiles,
detecta en cada tile y fusiona resultados. Diseñado para objetos pequeños y
densos (como alevines amontonados), donde la inferencia a imagen completa se
pierde detecciones.

Referencia: Akyon et al. (2022), "Slicing Aided Hyper Inference and Fine-tuning
for Small Object Detection", IEEE ICIP. https://arxiv.org/abs/2202.06934

Compara el conteo SAHI vs ground truth del test y reporta MAE/RMSE/MAPE, para
contrastar contra la inferencia estándar (evaluate_model.py).

Uso:
  python scripts/predict_sahi_counting.py --weights runs/detect/alevines_baseline_612/weights/best.pt \
      --slice 640 --overlap 0.2 --conf 0.3
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--test-images", default="data/dataset_yolo/images/test")
    p.add_argument("--manual-csv", default="data/counts/conteo_manual_test.csv")
    p.add_argument("--out-csv", default="reports/sahi_conteo.csv")
    p.add_argument("--slice", type=int, default=640, help="Tamaño de tile (px)")
    p.add_argument("--overlap", type=float, default=0.2, help="Solape entre tiles (0-1)")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    # sahi >=0.11 usa "ultralytics"; versiones previas "yolov8"
    det_model = None
    for mtype in ("ultralytics", "yolov8"):
        try:
            det_model = AutoDetectionModel.from_pretrained(
                model_type=mtype,
                model_path=args.weights,
                confidence_threshold=args.conf,
                device=args.device,
            )
            print(f"[SAHI] model_type='{mtype}' cargado.")
            break
        except Exception as e:
            print(f"[SAHI] model_type='{mtype}' falló: {e}")
    if det_model is None:
        raise SystemExit("[ERROR] No se pudo cargar el modelo en SAHI.")

    rows = list(csv.DictReader(open(args.manual_csv)))
    imgs = [(x["image_name"], int(x["real_count"])) for x in rows]
    tdir = Path(args.test_images)

    reals, preds, detail = [], [], []
    for name, real in imgs:
        result = get_sliced_prediction(
            str(tdir / name),
            det_model,
            slice_height=args.slice,
            slice_width=args.slice,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            verbose=0,
        )
        pred = len(result.object_prediction_list)
        reals.append(real); preds.append(pred)
        detail.append((name, real, pred, abs(real - pred)))
        print(f"  {name}: real={real}  sahi_pred={pred}  |err|={abs(real-pred)}")

    reals, preds = np.array(reals), np.array(preds)
    err = preds - reals
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((np.abs(err) / reals * 100).mean())
    bias = float(err.mean())
    agg_err = abs(int(preds.sum()) - int(reals.sum()))
    agg_pct = agg_err / int(reals.sum()) * 100

    print(f"\n===== SAHI (slice={args.slice}, overlap={args.overlap}, conf={args.conf}) =====")
    print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  bias={bias:+.2f}")
    print(f"  Lote: real={int(reals.sum())} pred={int(preds.sum())} "
          f"(error agregado {agg_err} = {agg_pct:.2f}%)")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "real_count", "sahi_pred", "error_abs"])
        w.writerows(detail)
        w.writerow([])
        w.writerow(["slice", args.slice, "overlap", args.overlap, "conf", args.conf])
        w.writerow(["MAE", round(mae, 2), "RMSE", round(rmse, 2),
                    "MAPE", round(mape, 2), "bias", round(bias, 2),
                    "agg_pct", round(agg_pct, 2)])
    print(f"  -> guardado en {out}")


if __name__ == "__main__":
    main()
