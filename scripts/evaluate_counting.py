"""
evaluate_counting.py
--------------------
Compara conteo manual vs conteo automático (YOLO) sobre un set de imágenes,
y reporta MAE, RMSE, MAPE y otras métricas relevantes para la tesis.

Espera un CSV de entrada con el conteo manual:
    data/counts/conteo_manual.csv
    columnas: image_name,real_count

Genera:
    data/counts/conteo_predicho.csv
        image_name, real_count, predicted_count, error_abs, error_pct
    reports/resultados_conteo.csv
        métricas agregadas (MAE, RMSE, MAPE, error_max, n_imagenes, etc.)
    reports/graficos/scatter_real_vs_pred.png
    reports/graficos/error_absoluto_por_imagen.png

Uso:
    python scripts/evaluate_counting.py --weights runs/detect/alevines_run/weights/best.pt
    python scripts/evaluate_counting.py --weights ... --images-dir data/raw/images --conf 0.3
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa conteo automático vs manual.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path al .pt entrenado")
    parser.add_argument("--manual-csv", type=str, default="data/counts/conteo_manual.csv",
                        help="CSV con conteo manual (columnas: image_name, real_count)")
    parser.add_argument("--images-dir", type=str, default="data/raw/images",
                        help="Carpeta donde buscar las imágenes listadas en el CSV")
    parser.add_argument("--predicted-csv", type=str, default="data/counts/conteo_predicho.csv",
                        help="CSV de salida con conteos predichos")
    parser.add_argument("--report-csv", type=str, default="reports/resultados_conteo.csv",
                        help="CSV de salida con métricas agregadas")
    parser.add_argument("--plots-dir", type=str, default="reports/graficos",
                        help="Carpeta para guardar gráficos")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def compute_metrics(real: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - real
    abs_err = np.abs(err)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # MAPE evitando división por cero (omite imágenes con real_count == 0)
    mask = real > 0
    if mask.sum() > 0:
        mape = float(np.mean(abs_err[mask] / real[mask]) * 100.0)
    else:
        mape = float("nan")

    # Sesgo (signed mean error): positivo = sobrecuenta, negativo = subcuenta
    bias = float(err.mean())

    return {
        "n_imagenes": int(len(real)),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE_%": round(mape, 4) if not math.isnan(mape) else None,
        "error_max": int(abs_err.max()) if len(abs_err) else 0,
        "error_min": int(abs_err.min()) if len(abs_err) else 0,
        "bias_promedio": round(bias, 4),
        "real_total": int(real.sum()),
        "pred_total": int(pred.sum()),
    }


def plot_results(real: np.ndarray, pred: np.ndarray, image_names: list[str],
                 plots_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Scatter real vs predicho con línea ideal y=x
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(real, pred, alpha=0.6, edgecolor="k")
    lim = max(real.max(), pred.max()) * 1.05 if len(real) else 1
    ax.plot([0, lim], [0, lim], "r--", label="y = x (ideal)")
    ax.set_xlabel("Conteo real (manual)")
    ax.set_ylabel("Conteo predicho (YOLO)")
    ax.set_title("Conteo manual vs automático")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "scatter_real_vs_pred.png", dpi=150)
    plt.close(fig)

    # 2) Error absoluto por imagen
    fig, ax = plt.subplots(figsize=(max(8, len(image_names) * 0.25), 5))
    abs_err = np.abs(pred - real)
    ax.bar(range(len(image_names)), abs_err)
    ax.set_xticks(range(len(image_names)))
    ax.set_xticklabels(image_names, rotation=90, fontsize=7)
    ax.set_ylabel("|error|")
    ax.set_title("Error absoluto por imagen")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "error_absoluto_por_imagen.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"[ERROR] Pesos no encontrados: {weights}")

    manual_csv = Path(args.manual_csv)
    if not manual_csv.exists():
        raise SystemExit(
            f"[ERROR] No existe el CSV de conteo manual: {manual_csv.resolve()}\n"
            "        Crealo con columnas: image_name,real_count"
        )

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"[ERROR] No existe la carpeta de imágenes: {images_dir.resolve()}")

    df_manual = pd.read_csv(manual_csv)
    required = {"image_name", "real_count"}
    if not required.issubset(df_manual.columns):
        raise SystemExit(f"[ERROR] {manual_csv} debe tener columnas: {required}")

    from ultralytics import YOLO
    model = YOLO(str(weights))

    print(f"Modelo: {weights}")
    print(f"Manual: {manual_csv}  ({len(df_manual)} filas)")
    print(f"Imágenes en: {images_dir.resolve()}\n")

    rows = []
    for _, row in df_manual.iterrows():
        img_name = str(row["image_name"]).strip()
        real_count = int(row["real_count"])
        img_path = images_dir / img_name

        if not img_path.exists():
            print(f"[AVISO] No se encontró: {img_path.name} — omitida")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[AVISO] No se pudo leer: {img_path.name} — omitida")
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
        pred_count = int(r.boxes.shape[0]) if r.boxes is not None else 0
        err_abs = abs(real_count - pred_count)
        err_pct = (err_abs / real_count * 100.0) if real_count > 0 else float("nan")

        rows.append({
            "image_name": img_name,
            "real_count": real_count,
            "predicted_count": pred_count,
            "error_abs": err_abs,
            "error_pct": round(err_pct, 4) if not math.isnan(err_pct) else None,
        })
        print(f"  {img_name}: real={real_count}  pred={pred_count}  |err|={err_abs}")

    if not rows:
        raise SystemExit("[ERROR] No se procesó ninguna imagen.")

    df_pred = pd.DataFrame(rows)
    predicted_csv = Path(args.predicted_csv)
    predicted_csv.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(predicted_csv, index=False)

    real = df_pred["real_count"].to_numpy()
    pred = df_pred["predicted_count"].to_numpy()
    metrics = compute_metrics(real, pred)

    print("\n=== Métricas agregadas ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    metrics["weights"] = str(weights)
    metrics["conf"] = args.conf
    metrics["iou"] = args.iou
    metrics["imgsz"] = args.imgsz

    report_csv = Path(args.report_csv)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    # Append-friendly: si existe, agregamos la fila; si no, creamos.
    df_report = pd.DataFrame([metrics])
    if report_csv.exists():
        df_existing = pd.read_csv(report_csv)
        df_report = pd.concat([df_existing, df_report], ignore_index=True)
    df_report.to_csv(report_csv, index=False)

    plot_results(real, pred, df_pred["image_name"].tolist(), Path(args.plots_dir))

    print(f"\nCSV de conteos: {predicted_csv}")
    print(f"CSV de métricas: {report_csv}")
    print(f"Gráficos:        {args.plots_dir}/")


if __name__ == "__main__":
    main()
