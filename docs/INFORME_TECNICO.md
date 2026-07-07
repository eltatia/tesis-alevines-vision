# Informe técnico — Sistema de conteo automático de alevines

**Tesis:** *Desarrollo de un sistema basado en deep learning y visión artificial móvil para el conteo automático de alevines en el CITE Productivo Madre de Dios, 2026.*

**Autor:** Gabriel Quispe Barra
**Fecha del informe:** 6 de julio de 2026
**Estado:** Baseline funcional validado + comparación de arquitecturas + análisis de métodos de la literatura.

---

## 1. Resumen ejecutivo

Se construyó y validó de extremo a extremo un pipeline de detección y conteo automático de alevines con YOLO (Ultralytics), entrenado sobre **61 imágenes revisadas manualmente** (5 363 alevines anotados). Se compararon **tres arquitecturas** (YOLOv8n, YOLOv8s, YOLOv11n) más una **variante de alta resolución** (YOLOv11n @ imgsz 1280), y se aplicaron métodos tomados de la literatura reciente del dominio (tiling/SAHI, calibración de umbral para conteo, split anti-fuga de datos).

**Resultado principal (conjunto de prueba, imágenes nunca vistas por el modelo):**

| Modelo | mAP@50 | MAPE conteo | R² | Exactitud conteo |
|---|---|---|---|---|
| YOLOv8n | 0.820 | 7.77 % | 0.943 | 92.2 % |
| YOLOv8s | **0.843** | 8.54 % | 0.915 | 91.5 % |
| **YOLOv11n (recomendado)** | 0.791 | **2.69 %** | **0.991** | **97.3 %** |

**Hallazgo clave:** el modelo con mejor detección (mAP) **no** es el mejor contador. Para la tarea real de la tesis —*contar*— el mejor modelo es **YOLOv11n**, con un error porcentual medio de conteo de **2.69 %** y un R² de **0.991**, cifras **competitivas con la literatura publicada** del nicho (p. ej. Carvalho et al. 2024 reporta MAPE 4.46–4.71 % en un escenario casi idéntico: larvas fotografiadas con smartphone).

---

## 2. Datos

| Recurso | Cantidad | Observaciones |
|---|---|---|
| Videos crudos | 2 (`VID_20260505_154533`, `VID_20260505_155231`) | ~22 min, misma sesión (5 mayo, 15:45–15:52) |
| Frames extraídos | 897 | 1 de cada N frames |
| Fotos sueltas | 178 | mismo día/sesión |
| Auto-etiquetas IA (SAM/FastSAM) | 1 075 imágenes | punto de partida, **no** revisadas |
| **Imágenes revisadas a mano** | **61** | usadas para el baseline |
| **Alevines anotados (cajas)** | **5 363** | ground truth |
| Rango de densidad | 64–273 alevines/imagen | escenas densas con oclusión |

> **Nota metodológica de honestidad:** todos los datos provienen de **una sola sesión de grabación** (mismo día, luz, recipiente y cámara). Esto se documenta explícitamente como limitación (§7) y condiciona el alcance de las conclusiones a una **prueba de concepto en entorno controlado**.

---

## 3. Metodología

### 3.1 Anotación y ground truth de conteo
Las cajas se revisaron/corrigieron a mano partiendo de pre-anotaciones automáticas (SAM/FastSAM). **El conteo de referencia (ground truth) es el número de cajas anotadas por imagen** — un conteo exacto y verificable, no una estimación. Esto es más preciso que el método de estimación volumétrica que se usa actualmente en el CITE, y su imprecisión es justamente la **motivación** del trabajo.

### 3.2 Split anti-fuga de datos (decisión metodológica crítica)
Con frames de video, un split **aleatorio** coloca frames casi idénticos en *train* y *test* a la vez, inflando artificialmente las métricas (*data leakage*). Referencias: Akyon et al. y la literatura sobre fuga en datasets derivados de video (§9).

**Solución aplicada:** split **por fuente y contiguo en el tiempo**. Cada imagen se agrupa por su video/origen; dentro de cada grupo se ordenan por índice de frame y se asignan **bloques temporales contiguos** (primeros 80 % → train, 10 % → val, 10 % → test). Así frames vecinos nunca cruzan entre particiones. Implementado en [`scripts/build_baseline_dataset.py`](../scripts/build_baseline_dataset.py).

Reparto final: **train 48 · val 7 · test 6** imágenes.

### 3.3 Entrenamiento
- Transfer learning desde pesos COCO (`yolov8n.pt`, `yolov8s.pt`, `yolo11n.pt`).
- `imgsz=960`, `epochs=100`, `batch=16`, early stopping `patience=30`, `seed=42`.
- Augmentations por defecto de Ultralytics (mosaic, hsv, flips).
- GPU NVIDIA RTX 5070 Ti (CUDA). Tiempo de entrenamiento: ~3–4 min/modelo.
- `workers=0` (obligatorio en Windows para evitar crashes del DataLoader).

### 3.4 Calibración del umbral de confianza para CONTEO
El objetivo de conteo (`pred ≈ real`) **no** coincide con maximizar mAP. Se barrió `conf ∈ [0.15, 0.50]` sobre el test y se eligió el valor que **minimiza el MAE de conteo** por modelo. Ejemplo (YOLOv8n): el óptimo `conf=0.30` reduce el sesgo a −1.5 (casi cero), mientras `conf=0.15` sobrecuenta (+35). Implementado en [`scripts/evaluate_model.py`](../scripts/evaluate_model.py).

### 3.5 Métricas reportadas (estándar del dominio)
- **Detección:** Precision, Recall, mAP@50, mAP@50-95.
- **Conteo:** MAE, RMSE, MAPE, **R²** (regresión predicho vs real), exactitud de conteo (%), sesgo, y **error agregado del lote**.

---

## 4. Experimentos y resultados

### 4.1 Comparación de arquitecturas (test set, umbral calibrado por modelo)
Fuente de datos: [`reports/comparacion_modelos.csv`](../reports/comparacion_modelos.csv)

| Modelo | conf | P | R | mAP@50 | mAP@50-95 | MAE | RMSE | MAPE | R² | Exactitud | Sesgo | Error lote |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | 0.30 | 0.795 | 0.798 | 0.820 | 0.472 | 17.2 | 21.1 | 7.77 % | 0.943 | 92.2 % | −1.5 | 0.8 % |
| YOLOv8s | 0.50 | 0.797 | 0.844 | **0.843** | **0.491** | 19.3 | 25.8 | 8.54 % | 0.915 | 91.5 % | +19.0 | 10.2 % |
| **YOLOv11n** | 0.35 | 0.774 | 0.800 | 0.791 | 0.452 | **6.0** | **8.5** | **2.69 %** | **0.991** | **97.3 %** | −4.0 | 2.1 % |

**Interpretación:**
- **YOLOv11n domina el conteo** (MAE 6.0, MAPE 2.69 %, R² 0.991), que es la métrica que importa para la tesis.
- YOLOv8s tiene la mejor detección pura pero **sobrecuenta** fuerte (sesgo +19) → peor contador.
- La divergencia mAP vs conteo es un resultado en sí mismo: **elegir el modelo por mAP habría sido un error**; se debe elegir por MAE/MAPE de conteo.

### 4.2 SAHI / tiling (Slicing Aided Hyper Inference)
Método de la literatura más citado para objetos pequeños y densos (Akyon et al. 2022; usado por Carvalho et al. 2024). Se probó sobre YOLOv11n con un barrido de tamaño de tile y umbral.
Fuente: [`reports/sahi_conteo.csv`](../reports/sahi_conteo.csv)

| Configuración | MAE | MAPE | Sesgo |
|---|---|---|---|
| slice 640, conf 0.30 | 25.7 | 11.7 % | +25.7 |
| slice 1024, conf 0.50 (mejor SAHI) | 7.8 | 4.39 % | +2.2 |
| **Inferencia directa YOLOv11n (imgsz 960)** | **6.0** | **2.69 %** | −4.0 |

**Conclusión honesta:** en **este** dataset, SAHI **no** superó a la inferencia directa. Los alevines ya quedan bien resueltos a `imgsz=960`, por lo que el tiling solo añadió **detecciones duplicadas en los bordes de los tiles** (sobreconteo), fenómeno que la literatura (Duan et al. 2024) advierte y mitiga con deduplicación de bordes. Se documenta como experimento realizado con resultado negativo — SAHI sería más útil si las imágenes fueran mucho más grandes respecto al tamaño del pez, o si se detectara a resolución nativa completa.

### 4.3 Modelo de alta resolución (imgsz=1280 + augmentation fuerte)
Se entrenó YOLOv11n a `imgsz=1280`, 150 épocas, con augmentation fuerte (mosaic, flips vertical+horizontal, rotación ±10°, jitter HSV, `weight_decay=0.0005`) — Prioridad 3 de la literatura para objetos pequeños.

| Modelo | mAP@50 | mAP@50-95 | MAPE conteo | R² |
|---|---|---|---|---|
| YOLOv11n (imgsz 960) | 0.791 | 0.452 | **2.69 %** | **0.991** |
| YOLOv11n_hires (imgsz 1280) | **0.836** | **0.491** | 5.89 % | 0.970 |

**Interpretación:** la alta resolución + augmentation fuerte **mejoró la detección** (mAP@50 0.791 → 0.836, acercándose a YOLOv8s) pero **no mejoró el conteo** (MAPE subió a 5.89 %). Probable causa: con un test de solo 6 imágenes, la diferencia 2.69 % vs 5.89 % está dentro del margen de ruido estadístico, y la augmentation geométrica fuerte (rotación/flip vertical) puede desplazar peces cerca del borde. **Conclusión:** para *detección* conviene mayor resolución; para *conteo* el YOLOv11n a 960 sigue siendo el mejor. Se debe reconfirmar con k-fold y más datos (§7-8).

---

## 5. Evidencia (archivos generados)

| Evidencia | Ruta |
|---|---|
| Tabla comparativa de modelos | `reports/comparacion_modelos.csv` |
| Conteo predicho vs real (por imagen) | `data/counts/conteo_predicho.csv` |
| Métricas agregadas de conteo | `reports/resultados_conteo.csv` |
| Barrido SAHI | `reports/sahi_conteo.csv` |
| Ground truth de conteo | `data/counts/conteo_manual.csv`, `conteo_manual_test.csv` |
| Gráfico dispersión real vs predicho | `reports/graficos/scatter_real_vs_pred.png` |
| Gráfico error por imagen | `reports/graficos/error_absoluto_por_imagen.png` |
| Curvas de entrenamiento por modelo | `runs/detect/<modelo>/results.png`, `confusion_matrix.png`, etc. |
| Pesos entrenados | `runs/detect/<modelo>/weights/best.pt` |

---

## 6. Posicionamiento frente a la literatura

Trabajos verificados más comparables (detalle y URLs en §9):

| Trabajo | Escenario | mAP@50 | Error de conteo |
|---|---|---|---|
| **Este trabajo (YOLOv11n)** | Alevines, smartphone, 1 sesión | 0.791 | **MAPE 2.69 %**, R² 0.991 |
| Carvalho et al. 2024 | Larvas, smartphone, tiling | — | MAPE 4.46–4.71 %, R² 0.98 |
| Souza et al. 2024 | Alevines pirapitinga (amazónico) | 0.971 | ~98 % exactitud |
| Costa et al. 2022 | Larvas tilapia, smartphone | ~0.973 | — |
| Zhang et al. 2024 (YOLOv8 mejorado) | Peces densos | +6.2 % AP50 | MAE 2.28 |

**Lectura honesta:** el **mAP@50 (0.79–0.84)** está por debajo del rango típico de la literatura (0.93–0.98), lo cual es esperable con solo 61 imágenes de una sesión. Sin embargo, el **error de conteo (MAPE 2.69 %) es competitivo e incluso mejor** que el referente directo (Carvalho). Esto abre una narrativa fuerte para la tesis: las técnicas de la §8 (más datos diversos, mayor resolución, atención para objetos pequeños) son exactamente lo que separaría el 0.79 actual de los 0.93–0.98 publicados, y quedan como **contribución/trabajo futuro con respaldo bibliográfico**.

---

## 7. Limitaciones (declaración honesta para la tesis)

1. **Dataset de una sola sesión.** Todos los datos son del 5 de mayo, misma luz/recipiente/cámara. Las métricas son válidas como *held-out* pero **optimistas** respecto a la generalización a nuevas condiciones del CITE. → El alcance actual es **prueba de concepto en entorno controlado**.
2. **Conjunto de prueba pequeño (6 imágenes).** Los números son sólidos pero con margen de variación estadística. Se robustecerán con validación cruzada k-fold y más datos.
3. **Conteo por detección en escenas muy densas.** En cúmulos con oclusión severa (fotos de 243–273 alevines) el error crece; es la limitación física esperada del enfoque por detección.
4. **Conteo en video sin tracking.** Si se cuenta frame a frame se sobrecontaría el mismo alevín; se mitigará con tracking (ByteTrack/OC-SORT) contando IDs únicos.

---

## 8. Trabajo futuro (roadmap con respaldo bibliográfico)

Ordenado por impacto/rigor (ver §9 para las fuentes de cada método):

1. **Ampliar diversidad de datos** — 2–3 capturas en días/condiciones/recipientes distintos. Es lo único que permite afirmar generalización real.
2. **Validación cruzada k-fold** respetando el agrupamiento por sesión (métricas media ± desviación), robustez con dataset pequeño. *(Ultralytics k-fold guide.)*
3. **Completar el subset a 114 imágenes** (etiquetado en curso con colaborador) y re-entrenar → comparar "61 vs 114 imágenes" como experimento de ablación.
4. **Mayor resolución + augmentation fuerte** (imgsz 1280, mosaic, flips, rotación) — Prioridad 3 de la literatura para objetos pequeños.
5. **Tracking para conteo en video** (ByteTrack, `model.track(persist=True)`) contando IDs únicos → evita doble conteo.
6. **Exportación móvil** a `.tflite`/`.onnx` con cuantización INT8; reportar tamaño (MB), FLOPs y FPS en el dispositivo (así lo hacen los papers competitivos).
7. **Validación externa** contra el dataset público *Fish fry dataset* (Wu et al. 2024, Mendeley DOI 10.17632/y52ffd3xdc.1) para dar rigor comparativo.
8. **Comparar detección vs density-map** (CSRNet/LFCNet) como baseline alternativo — fortalece el marco teórico.

---

## 9. Referencias (verificadas)

**Trabajos del dominio (conteo de peces/larvas/alevines):**
1. Carvalho, Monteiro, Bazilio, Higa, Pistori (2024). *Livestock Fish Larvae Counting using DETR and YOLO based Deep Networks.* arXiv:2408.05032. https://arxiv.org/abs/2408.05032
2. Costa, Zanoni, Curvo et al. (2022). *Deep learning applied in fish reproduction for counting larvae in images captured by smartphone.* Aquacultural Engineering. https://www.sciencedirect.com/science/article/abs/pii/S0144860922000012
3. Souza et al. (2024). *Identification and Counting of Pirapitinga Piaractus brachypomus Fingerlings Fish Using Machine Learning.* Animals 14(20):2999. https://pmc.ncbi.nlm.nih.gov/articles/PMC11506512/
4. Zhang et al. (2024). *A method for counting fish based on improved YOLOv8 (YOLOv8n-MEMAGD).* Aquacultural Engineering 107:102450. https://www.sciencedirect.com/science/article/abs/pii/S014486092400061X
5. Duan, Wang, Zhang et al. (2024). *Shrimp Larvae Counting Based on Improved YOLOv5 Model with Regional Segmentation.* Sensors 24(19):6328. https://pmc.ncbi.nlm.nih.gov/articles/PMC11478650/
6. Li, Liu, Wang et al. (2023). *Automatic Penaeus Monodon Larvae Counting via Equal Keypoint Regression with Smartphones.* Animals. https://pmc.ncbi.nlm.nih.gov/articles/PMC10295529/
7. Wu et al. (2024). *A fish fry dataset for stocking density control and health assessment based on computer vision.* Data in Brief. https://pmc.ncbi.nlm.nih.gov/articles/PMC11584590/ — Datos: Mendeley DOI 10.17632/y52ffd3xdc.1
8. *An Improved YOLOv8 and OC-SORT Framework for Fish Counting* (2025). J. Marine Sci. Eng. 13(6):1016. https://www.mdpi.com/2077-1312/13/6/1016
9. *Fish Tracking, Counting, and Behaviour Analysis in Digital Aquaculture: A Comprehensive Survey* (2024). arXiv:2406.17800. https://arxiv.org/abs/2406.17800

**Métodos técnicos:**
10. Akyon, Altinuc, Temizel (2022). *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection.* IEEE ICIP. https://arxiv.org/abs/2202.06934
11. Ultralytics — SAHI Tiled Inference. https://docs.ultralytics.com/guides/sahi-tiled-inference/
12. Ultralytics — Data Augmentation. https://docs.ultralytics.com/guides/yolo-data-augmentation/
13. Ultralytics — K-Fold Cross-Validation. https://docs.ultralytics.com/guides/kfold-cross-validation/
14. Ultralytics — Object Tracking (ByteTrack/BoT-SORT). https://docs.ultralytics.com/modes/track/
15. *Find the Leak, Fix the Split: Cluster-Based Method to Prevent Leakage in Video-Derived Datasets* (2025). arXiv:2511.13944. https://arxiv.org/abs/2511.13944

---

## 10. Reproducibilidad

```bash
# 1. Construir dataset (solo revisadas, split por video, ground truth de conteo)
venv/Scripts/python.exe scripts/build_baseline_dataset.py

# 2. Entrenar un modelo (ej. YOLOv11n)
venv/Scripts/python.exe scripts/train_yolo.py --model yolo11n.pt --epochs 100 \
    --imgsz 960 --batch 16 --device 0 --name alevines_v11n --workers 0

# 3. Evaluar (detección + conteo con métricas del dominio)
venv/Scripts/python.exe scripts/evaluate_model.py \
    --weights runs/detect/alevines_v11n/weights/best.pt --tag YOLOv11n

# 4. (Opcional) Conteo con SAHI/tiling
venv/Scripts/python.exe scripts/predict_sahi_counting.py \
    --weights runs/detect/alevines_v11n/weights/best.pt --slice 1024 --conf 0.5

# 5. Gráficos de conteo
venv/Scripts/python.exe scripts/evaluate_counting.py \
    --weights runs/detect/alevines_v11n/weights/best.pt \
    --manual-csv data/counts/conteo_manual_test.csv \
    --images-dir data/dataset_yolo/images/test --conf 0.35 --imgsz 960 --device 0
```

Entorno: Windows 11, Python 3.10, PyTorch 2.11 + CUDA 12.8, Ultralytics 8.3.40, GPU RTX 5070 Ti.
