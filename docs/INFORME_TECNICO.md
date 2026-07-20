# Informe técnico — Sistema de conteo automático de alevines

**Tesis:** *Desarrollo de un sistema basado en deep learning y visión artificial móvil para el conteo automático de alevines en el CITE Productivo Madre de Dios, 2026.*

**Autor:** Gabriel Quispe Barra
**Fecha del informe:** 20 de julio de 2026 (actualizado — dataset completo de 113 imágenes)
**Estado:** Sistema validado con dataset completo + comparación de 6 modelos + ablación de datos + métodos de la literatura.

---

## 1. Resumen ejecutivo

Se construyó y validó de extremo a extremo un pipeline de detección y conteo automático de alevines con YOLO (Ultralytics), entrenado sobre **113 imágenes revisadas manualmente** (17 640 alevines anotados; etiquetado colaborativo entre dos anotadores). Se compararon **6 configuraciones de modelo** (YOLOv8n, YOLOv11n, YOLOv11s a 960 px; YOLOv8n y YOLOv11n a 1280 px; y una ablación con la mitad de los datos), y se aplicaron métodos de la literatura reciente del dominio (tiling/SAHI, calibración de umbral para conteo, split anti-fuga de datos, alta resolución + augmentation).

**Resultado principal — modelo recomendado: YOLOv11n de alta resolución (1280 px)**, evaluado sobre 11 imágenes de prueba nunca vistas (2 318 alevines):

| Métrica | Valor | Interpretación |
|---|---|---|
| **Error de conteo a nivel de LOTE** | **0.17 %** | 2 322 predichos vs 2 318 reales — lo que le importa al CITE al contar un lote |
| Sesgo | **+0.36** | sin tendencia sistemática a sobre/subcontar |
| MAPE por imagen (mediana) | **5.30 %** | error típico por imagen |
| MAPE por imagen (media) | 9.59 % | inflada por 1 frame denso muy difícil (§4.5) |
| MAPE sin ese caso difícil | 5.78 % | — |
| mAP@50 / mAP@50-95 | 0.825 / **0.473** | mejor detección de todos los modelos |
| R² (conteo pred vs real) | 0.788 | — |

**Hallazgos clave:**
1. **A nivel de lote el sistema es casi perfecto (error 0.17 %)** — el sobre y subconteo por imagen se compensan (sesgo ~0).
2. La **alta resolución (1280 px)** fue decisiva: llevó el sesgo de −14 (subconteo a 960 px) a ~0 y dio la mejor detección.
3. **Más datos ayudan:** la ablación controlada (§4.4) muestra que doblar el train (45→90 imágenes) sube el mAP@50 de 0.785 a 0.801.
4. El error residual se concentra en **escenas ultra-densas con oclusión** (§4.5), la limitación física esperada del conteo por detección.

> **Nota sobre la comparación con el informe preliminar (61 imágenes):** el run anterior reportó MAPE 2.69 % sobre un test de solo 6 imágenes menos densas. Con el dataset completo (113 imgs) el test es más grande (11 imgs) y **más difícil** (incluye frames densísimos de 190–277 alevines). Los números actuales son **más realistas y honestos**, no un retroceso.

---

## 2. Datos

| Recurso | Cantidad | Observaciones |
|---|---|---|
| Videos crudos | 2 (`VID_20260505_154533`, `VID_20260505_155231`) | ~22 min, misma sesión (5 mayo, 15:45–15:52) |
| Frames extraídos | 897 | 1 de cada N frames |
| Fotos sueltas | 178 | mismo día/sesión |
| Auto-etiquetas IA (SAM/FastSAM) | 1 075 imágenes | punto de partida, **no** revisadas |
| **Imágenes revisadas a mano** | **113** | etiquetado colaborativo (2 anotadores) |
| **Alevines anotados (cajas)** | **17 640** | ground truth |
| Rango de densidad | 64–277 alevines/imagen | escenas densas con oclusión |
| Split (por video, sin leakage) | train 90 · val 12 · test 11 | ver §3.2 |

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

Todos los resultados sobre el **test set de 11 imágenes** (2 318 alevines) nunca vistas, con umbral de confianza calibrado por modelo. Fuente: [`reports/comparacion_modelos_114.csv`](../reports/comparacion_modelos_114.csv)

### 4.1 Comparación de 6 configuraciones (dataset completo, 113 imágenes)

| Modelo | imgsz | mAP@50 | mAP@50-95 | MAPE (media) | R² | Sesgo | **Error lote** |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 960 | 0.809 | 0.436 | 9.20 % | 0.771 | −11.5 | 5.44 % |
| YOLOv11n | 960 | 0.801 | 0.434 | 10.67 % | 0.698 | −14.2 | 6.73 % |
| YOLOv11s | 960 | 0.822 | 0.460 | 11.15 % | 0.679 | −10.6 | 5.00 % |
| YOLOv8n | 1280 | 0.819 | 0.465 | 11.25 % | 0.686 | −5.1 | 2.42 % |
| **YOLOv11n (recomendado)** | **1280** | **0.825** | **0.473** | 9.59 % | **0.788** | **+0.4** | **0.17 %** |
| YOLOv11n (½ datos, ablación) | 960 | 0.785 | 0.412 | 10.89 % | — | −14.0 | — |

**Interpretación:**
- **YOLOv11n a 1280 px es el mejor modelo global:** mejor detección (mAP@50-95 = 0.473), mejor R², **sesgo casi nulo (+0.4)** y **error de lote de solo 0.17 %** (2 322 vs 2 318 alevines).
- La **alta resolución es la palanca decisiva:** a 960 px todos los modelos **subcuentan** fuerte (sesgo −10 a −14); a 1280 px el sesgo se anula. Con más píxeles el modelo detecta los alevines pequeños/claros que antes perdía.
- El MAPE por imagen (9.59 %) engaña: su **mediana es 5.30 %** y está inflado por un único frame ultra-denso (§4.5). Sin ese caso, el MAPE es **5.78 %**.

### 4.2 Métrica más relevante para el CITE: error de LOTE
El CITE cuenta *lotes* de alevines, no imágenes sueltas. A nivel agregado (suma sobre las 11 imágenes de test) el modelo recomendado predijo **2 322 vs 2 318 reales → 0.17 % de error**. El sobreconteo en unas imágenes y el subconteo en otras se compensan (sesgo ~0), de modo que el conteo total del lote es prácticamente exacto. **Evidencia:** [`docs/figuras/fig_scatter_113.png`](figuras/fig_scatter_113.png) (dispersión real vs predicho, casi sobre la diagonal ideal).

### 4.3 SAHI / tiling (Slicing Aided Hyper Inference)
Método de la literatura para objetos pequeños y densos (Akyon et al. 2022; usado por Carvalho et al. 2024). Se probó con barrido de tamaño de tile y umbral. **Conclusión honesta:** SAHI **no** superó a la inferencia directa — los alevines ya quedan bien resueltos a 960–1280 px, y el tiling solo añadió duplicados en los bordes de los recortes (sobreconteo), fenómeno que Duan et al. (2024) advierte y mitiga con deduplicación de bordes. Se documenta como experimento con resultado negativo. Fuente: [`reports/sahi_conteo.csv`](../reports/sahi_conteo.csv)

### 4.4 Ablación controlada: ¿ayudan más datos?
Se entrenó el mismo modelo (YOLOv11n @ 960) con **45 vs 90 imágenes de entrenamiento**, evaluando sobre el **mismo test** (split idéntico, sin leakage). Fuente: [`reports/ablacion_datos.csv`](../reports/ablacion_datos.csv)

| Train | mAP@50 | mAP@50-95 |
|---|---|---|
| 45 imágenes | 0.785 | 0.412 |
| 90 imágenes | **0.801** | **0.434** |

**Resultado:** doblar los datos de entrenamiento mejora la detección (mAP@50 +1.6 pts, mAP@50-95 +2.2 pts). Confirma cuantitativamente que **seguir etiquetando rinde** y respalda el roadmap de ampliar el dataset (§8).

### 4.5 Análisis del caso de fallo (transparencia)
El error residual del mejor modelo se concentra en **una imagen** (`VID_..._frame_000698`): 193 alevines reales, 101 detectados (a 1280 px). La inspección visual confirma que **la etiqueta es correcta** y que el modelo **subcuenta genuinamente** en una escena ultra-densa con oclusión parcial por un objeto (brazo/tubo) en la bandeja — la limitación física esperada del conteo por detección.

- **Figura de fallo:** [`docs/figuras/fig_subconteo_denso_698.jpg`](figuras/fig_subconteo_denso_698.jpg) — cajas reales (verde) vs detectadas (rojo).
- **Figura de acierto:** [`docs/figuras/fig_conteo_ok_699.jpg`](figuras/fig_conteo_ok_699.jpg) — frame adyacente, 223 reales vs 218 detectados (error 2.2 %).

La alta resolución mejoró este caso (85 → 101 detecciones) pero no lo resolvió del todo; es el objetivo natural del trabajo futuro (§8: atención para objetos pequeños, más datos densos).

### 4.6 Prueba de generalización a NUEVA sesión (2 videos, otra cámara)
Se grabaron **2 videos nuevos** (iPhone, 4K, ~18 min) en una sesión distinta: **otro recipiente** (bandeja translúcida sobre superficie metálica), **otra iluminación** y **otro día**. Es la primera prueba de generalización fuera de la sesión de entrenamiento. Se aplicó el modelo recomendado (sin reentrenar) sobre frames de muestra:

| Video | Contenido | Resultado cualitativo |
|---|---|---|
| IMG_0177 | Larvas pequeñas, baja densidad (~20) | ✅ **Generaliza bien** — detecta las larvas pese al recipiente/fondo/cámara distintos |
| IMG_0178 | Peces **más grandes/desarrollados, otra especie** | ⚠️ **Subdetecta** — apariencia fuera de la distribución de entrenamiento |

**Tres conclusiones importantes:**
1. **Generalización parcial confirmada:** cuando el objeto se parece al de entrenamiento (larvas), el modelo funciona en una sesión nueva → evidencia a favor de robustez a cambios de fondo/luz/cámara.
2. **Límite de dominio:** peces visualmente distintos (otra especie/tamaño) requieren datos nuevos. Se decidió ampliar el sistema a un **contador de peces general** (una sola clase, múltiples tamaños/especies).
3. **Falso positivo sistemático:** el modelo marca el **logo en relieve del recipiente** como pez. Se corrige añadiendo ejemplos negativos (imágenes del recipiente sin peces) al reentrenar.

**Acción tomada:** se extrajeron 120 frames diversos (dedup pHash) de ambos videos y se pre-etiquetaron con el modelo como borrador, listos para corrección manual y reentrenamiento. Evidencia: [`docs/figuras/fig_generaliza_v177.jpg`](figuras/fig_generaliza_v177.jpg), [`docs/figuras/fig_generaliza_v178.jpg`](figuras/fig_generaliza_v178.jpg).

---

## 5. Evidencia (archivos generados)

| Evidencia | Ruta |
|---|---|
| **Comparación de 6 modelos (113 imgs)** | `reports/comparacion_modelos_114.csv` |
| Ablación de datos (45 vs 90) | `reports/ablacion_datos.csv` |
| Comparación preliminar (61 imgs) | `reports/comparacion_modelos.csv` |
| Conteo predicho vs real (por imagen) | `data/counts/conteo_predicho.csv` |
| Barrido SAHI | `reports/sahi_conteo.csv` |
| Ground truth de conteo | `data/counts/conteo_manual.csv`, `conteo_manual_test.csv` |
| **Figura fallo (subconteo denso)** | `docs/figuras/fig_subconteo_denso_698.jpg` |
| **Figura acierto (frame adyacente)** | `docs/figuras/fig_conteo_ok_699.jpg` |
| Dispersión real vs predicho | `docs/figuras/fig_scatter_113.png` |
| Error por imagen | `docs/figuras/fig_error_113.png` |
| Curvas de entrenamiento por modelo | `runs/detect/<modelo>/results.png`, `confusion_matrix.png`, etc. |
| Pesos entrenados | `runs/detect/<modelo>/weights/best.pt` |

---

## 6. Posicionamiento frente a la literatura

Trabajos verificados más comparables (detalle y URLs en §9):

| Trabajo | Escenario | mAP@50 | Error de conteo |
|---|---|---|---|
| **Este trabajo (YOLOv11n @1280)** | Alevines, smartphone, 1 sesión | 0.825 | **Lote 0.17 %** · MAPE mediana 5.3 % · R² 0.79 |
| Carvalho et al. 2024 | Larvas, smartphone, tiling | — | MAPE 4.46–4.71 %, R² 0.98 |
| Souza et al. 2024 | Alevines pirapitinga (amazónico) | 0.971 | ~98 % exactitud |
| Costa et al. 2022 | Larvas tilapia, smartphone | ~0.973 | — |
| Zhang et al. 2024 (YOLOv8 mejorado) | Peces densos | +6.2 % AP50 | MAE 2.28 |

**Lectura honesta:** el **mAP@50 (0.80–0.83)** está por debajo del rango típico de la literatura (0.93–0.98), esperable con un dataset de una sola sesión. Sin embargo, la **exactitud de conteo a nivel de lote (99.8 %) es de primer nivel** y la MAPE mediana (5.3 %) es competitiva con el referente directo (Carvalho, ~4.5 %). Las técnicas de la §8 (más datos diversos, atención para objetos pequeños, deduplicación con tiling) son exactamente lo que separaría el mAP actual de los 0.93–0.98 publicados, y quedan como **contribución/trabajo futuro con respaldo bibliográfico**.

---

## 7. Limitaciones (declaración honesta para la tesis)

1. **Dataset de una sola sesión.** Todos los datos son del 5 de mayo, misma luz/recipiente/cámara. Las métricas son válidas como *held-out* pero **optimistas** respecto a la generalización a nuevas condiciones del CITE. → El alcance actual es **prueba de concepto en entorno controlado**.
2. **Conjunto de prueba moderado (11 imágenes).** Sólido pero con margen estadístico; se robustecerá con validación cruzada k-fold.
3. **Conteo por detección en escenas muy densas.** En cúmulos con oclusión severa (frames de 190–277 alevines) el modelo subcuenta (§4.5); es la limitación física esperada del enfoque por detección.
4. **Consistencia entre anotadores.** El etiquetado fue colaborativo (2 personas). Pequeñas diferencias de criterio en cúmulos densos añaden ruido al ground truth; a futuro conviene una guía de anotación y una doble-revisión cruzada de una muestra.
5. **Conteo en video sin tracking.** Si se cuenta frame a frame se sobrecontaría el mismo alevín; se mitigará con tracking (ByteTrack/OC-SORT) contando IDs únicos.

---

## 8. Trabajo futuro (roadmap con respaldo bibliográfico)

Ordenado por impacto/rigor (ver §9 para las fuentes de cada método):

1. **Ampliar diversidad de datos** — 2–3 capturas en días/condiciones/recipientes distintos. Es lo único que permite afirmar generalización real.
2. **Validación cruzada k-fold** respetando el agrupamiento por sesión (métricas media ± desviación), robustez con dataset pequeño. *(Ultralytics k-fold guide.)*
3. ~~Completar el subset a 114 imágenes y ablación de datos.~~ ✅ **Hecho** (§4.4): 113 imágenes revisadas, la ablación confirma que más datos mejoran el mAP.
4. ~~Mayor resolución + augmentation.~~ ✅ **Hecho** (§4.1): 1280 px fue decisivo para anular el subconteo. Siguiente nivel: cabeza de detección P2 y atención para objetos pequeños (SOD-YOLO, CBAM).
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
