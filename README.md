# Sistema de conteo automático de alevines

**Tesis:** Desarrollo de un sistema basado en deep learning y visión artificial móvil para el conteo automático de alevines en el CITE Productivo Madre de Dios, 2026.

Sistema basado en YOLO (Ultralytics) + OpenCV para detectar y contar alevines en imágenes y videos capturados con celular.

---

## Estructura del proyecto

```
TESIS/
├── data/
│   ├── raw/{images,videos}/         # Datos originales (NO modificar)
│   ├── frames/{all,selected}/       # Frames extraídos de videos
│   ├── dataset_yolo/                # Dataset listo para YOLO
│   │   ├── images/{train,val,test}/
│   │   ├── labels/{train,val,test}/
│   │   └── data.yaml
│   └── counts/                      # CSV con conteos manuales y predichos
├── notebooks/                       # Análisis exploratorio (opcional)
├── scripts/                         # Scripts principales
│   ├── extract_frames.py
│   ├── split_dataset.py
│   ├── train_yolo.py
│   ├── predict_image.py
│   ├── predict_video.py
│   └── evaluate_counting.py
├── models/{yolov8n,yolov8s,exports}/
├── runs/                            # Salida automática de Ultralytics
├── reports/
│   ├── imagenes_resultado/          # Imágenes con cajas dibujadas
│   ├── graficos/                    # PNG de métricas
│   ├── metricas_modelo.csv
│   └── resultados_conteo.csv
├── app/futura_app_movil/            # Placeholder app móvil (Fase 14+)
├── requirements.txt
└── README.md
```

---

## 1. Instalación

Requiere **Python 3.10** en Windows. Desde la raíz del proyecto:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para usar GPU NVIDIA (recomendado si entrenas más de 50 épocas):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. Colocar los datos

- Imágenes originales (`.jpg`/`.png`) → `data/raw/images/`
- Videos originales (`.mp4`/`.mov`/...) → `data/raw/videos/`

No edites los archivos originales. Todo el trabajo se hace sobre copias o frames extraídos.

---

## 3. Extraer frames desde los videos

```powershell
# Por defecto: 1 frame cada 30, sin deduplicación
python scripts/extract_frames.py

# Recomendado: con deduplicación perceptual para evitar etiquetar duplicados
python scripts/extract_frames.py --interval 30 --dedup --phash-threshold 5
```

Salida: `data/frames/all/<videoname>_frame_NNNNNN.jpg`

---

## 4. Seleccionar frames útiles (manual)

Revisa `data/frames/all/` y copia los frames con buena iluminación, alevines visibles y variedad de densidad a `data/frames/selected/`.

Criterios de descarte: imagen borrosa, agua muy turbia, reflejos fuertes, alevines no visibles, duplicados.

> Sugerencia: combina las 257 imágenes originales (`data/raw/images/`) con `data/frames/selected/` antes de etiquetar, para máxima variedad.

---

## 5. Etiquetar (Roboflow / LabelImg / CVAT)

- **Clase única:** `alevin` (id `0`)
- **Formato:** YOLO (`class_id x_center y_center w h`, todo normalizado 0-1)
- Cada `img_001.jpg` debe tener su `img_001.txt` en la misma carpeta (o en otra que pasarás a `split_dataset.py` con `--labels-src`)

---

## 6. Dividir en train/val/test

```powershell
# Default: 80/10/10, copiando archivos
python scripts/split_dataset.py

# Custom: 70/20/10, moviendo en vez de copiar
python scripts/split_dataset.py --train 0.7 --val 0.2 --test 0.1 --move
```

Verifica `data/dataset_yolo/data.yaml`. La ruta `path:` es relativa al script de entrenamiento.

---

## 7. Entrenar YOLO

```powershell
# Primer entrenamiento (YOLOv8n, CPU)
python scripts/train_yolo.py --model yolov8n.pt --epochs 100 --imgsz 640 --batch 8

# Con GPU
python scripts/train_yolo.py --model yolov8n.pt --epochs 100 --device 0

# Comparar con YOLOv8s o YOLOv11n
python scripts/train_yolo.py --model yolov8s.pt --name alevines_yolov8s
python scripts/train_yolo.py --model yolo11n.pt --name alevines_yolo11n
```

Salida: `runs/detect/<name>/weights/best.pt`

Validar después del entrenamiento:

```powershell
yolo detect val model=runs/detect/alevines_run/weights/best.pt data=data/dataset_yolo/data.yaml
```

---

## 8. Predecir en imágenes

```powershell
# Una imagen
python scripts/predict_image.py --weights runs/detect/alevines_run/weights/best.pt --source data/raw/images/img_001.jpg

# Carpeta completa
python scripts/predict_image.py --weights runs/detect/alevines_run/weights/best.pt --source data/dataset_yolo/images/test --conf 0.3
```

Resultados con cajas dibujadas en `reports/imagenes_resultado/`.

---

## 9. Predecir en video

```powershell
python scripts/predict_video.py --weights runs/detect/alevines_run/weights/best.pt --source data/raw/videos/video01.mp4
```

> ⚠️ **Limitación v1:** el conteo se hace por frame de forma independiente, sin tracking. El mismo alevín se cuenta varias veces a lo largo del video. Documentar como limitación en la tesis y proponer ByteTrack/BoT-SORT como trabajo futuro.

---

## 10. Evaluar conteo automático vs manual

Crea `data/counts/conteo_manual.csv`:

```csv
image_name,real_count
img_001.jpg,45
img_002.jpg,62
img_003.jpg,38
```

Luego:

```powershell
python scripts/evaluate_counting.py --weights runs/detect/alevines_run/weights/best.pt --images-dir data/raw/images
```

Genera:
- `data/counts/conteo_predicho.csv` — fila por imagen con error absoluto y porcentual
- `reports/resultados_conteo.csv` — MAE, RMSE, MAPE, bias, etc. (acumulativo entre corridas)
- `reports/graficos/scatter_real_vs_pred.png` y `error_absoluto_por_imagen.png`

---

## 11. Exportar a móvil (Fase 14)

```powershell
# ONNX (universal)
yolo export model=runs/detect/alevines_run/weights/best.pt format=onnx

# TFLite (Android)
pip install tensorflow==2.17.0
yolo export model=runs/detect/alevines_run/weights/best.pt format=tflite
```

Los archivos exportados quedan junto al `best.pt`. Copiar al destino: `models/exports/`.

---

## Métricas que debes reportar en la tesis

**De detección** (las da `yolo detect val`):
- Precision, Recall, F1-score
- mAP@50, mAP@50-95

**De conteo** (las da `evaluate_counting.py`):
- MAE (error absoluto promedio)
- RMSE (raíz del error cuadrático medio)
- MAPE (% de error promedio)
- Sesgo (bias) — si > 0, sobrecuenta; si < 0, subcuenta
- Error máximo y mínimo

---

## Estado del proyecto (mayo 2026)

- [x] Estructura del proyecto + scripts base
- [ ] Colocar 257 imágenes en `data/raw/images/`
- [ ] Colocar 5 videos en `data/raw/videos/`
- [ ] Extraer y filtrar frames
- [ ] Etiquetar dataset
- [ ] Primer entrenamiento YOLOv8n
- [ ] Evaluación de detección y conteo
- [ ] Comparación entre arquitecturas (v8n, v8s, v11n)
- [ ] Exportación a TFLite
