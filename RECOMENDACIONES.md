# Recomendaciones del asistente — proyecto de tesis alevines

**Fecha:** 2026-05-17
**Tesis:** Desarrollo de un sistema basado en deep learning y visión artificial móvil para el conteo automático de alevines en el CITE Productivo Madre de Dios, 2026.

Este archivo concentra las recomendaciones, riesgos y decisiones discutidas con el asistente para no perderlas entre conversaciones. Cualquier cambio importante debería actualizarse aquí.

---

## 1. Análisis de la guía técnica de 16 fases

### Lo que está sólido y se mantiene

- **Estructura de carpetas** estándar para proyectos YOLO (`raw → frames → dataset_yolo` con `images/labels` divididos en train/val/test). Ya creada en este proyecto.
- **Orden de fases:** datos → etiquetado → entrenamiento → evaluación. Lógico y reusable.
- **Separación entre métricas de detección y métricas de conteo.** Es la decisión metodológica correcta para una tesis: una cosa es "¿el modelo detecta bien?" (Precision/Recall/mAP) y otra es "¿cuenta bien?" (MAE/RMSE/MAPE).
- **Elección de YOLOv8n como modelo base.** Acertada porque el destino final es móvil. Empezar liviano y subir solo si las métricas no alcanzan.
- **Exportación a `.tflite` / `.onnx`** prevista desde el inicio para la futura app móvil.

### Riesgos detectados y cómo se mitigaron en los scripts

| Riesgo | Mitigación implementada |
|---|---|
| Dataset chico para YOLO desde cero | Transfer learning desde `yolov8n.pt`. Augmentations YOLO por defecto activas (mosaic, mixup, hsv, flip). |
| Frames extraídos cada 15 = miles de casi duplicados | `extract_frames.py` con intervalo default **30** + flag `--dedup` con pHash. |
| Etiquetar 200-500 imágenes con docenas de alevines es trabajo pesado | Estrategia de **pseudo-labeling** después del primer modelo (ver sección 4). |
| Conteo por frame en video sobrecuenta sin tracking | `predict_video.py` documenta la limitación. Para v1 reportar conteo solo sobre imágenes estáticas. v2 integrar ByteTrack. |
| `tensorflow` no estaba en `requirements.txt` para exportar a TFLite | Documentado en `requirements.txt` como instalación opcional separada. |
| Falta de reproducibilidad | `train_yolo.py` y `split_dataset.py` usan `seed=42` fija. Versiones de librerías pineadas en `requirements.txt`. |
| YOLOv11 mencionado en la guía pero comandos con v8 | `train_yolo.py` acepta cualquier checkpoint (`yolo11n.pt`, `yolo11s.pt`, etc.). |
| Densidad alta de alevines = oclusiones | Si `mAP50 < 0.70` en el primer entrenamiento, escalar a `imgsz=800` o tiling (sección 6). |
| Split 70/20/10 deja test set pequeño | Default cambiado a **80/10/10**. Considerar k-fold para reportes en la tesis si test queda muy chico. |

---

## 2. Decisiones tomadas en esta sesión

- Proyecto creado **directamente en** `c:\Users\Usuario\TESIS\` (sin subcarpeta `alevines_vision_tesis/`).
- **Git inicializado** en rama `main` + `.gitignore` que excluye `data/`, `models/`, `runs/`, `venv/`, pesos y exports.
- **Intervalo de extracción de frames:** 30 frames + deduplicación pHash activable (también se probó 15 con éxito).
- **Split por defecto:** 80/10/10 con seed 42.
- **Modelo inicial:** YOLOv8n con `epochs=100`, `imgsz=1280`, `batch=8`, `device=cuda`, `patience=30`. Se subió imgsz de 640 a 1280 por el tamaño relativo pequeño de los alevines en las imágenes originales (4-12 MP).
- **GPU detectada:** RTX 5070 Ti (Blackwell, sm_120, 16 GB VRAM). PyTorch 2.11.0+cu128 instalado (¡cuidado!, el índice cu121 estándar NO funciona con Blackwell).
- **Estrategia de etiquetado:** **auto-labeling con SAM 2 + revisión manual** en LabelImg. SAM 2 segmenta todo el contenido y filtramos por área/aspect-ratio para quedarnos solo con alevines.
- **Reportes acumulativos:** `reports/resultados_conteo.csv` agrega una fila por corrida (no sobrescribe).

## 2.1 Inventario real de datos (auditoría con `inspect_data.py`)

**Imágenes:** 250 archivos JPG, 1.23 GB. 3 resoluciones (post-EXIF): 3072×4096 (153), 4592×8160 (96), 2160×3840 (1). Todas verticales. **Ráfagas detectadas: 161 imágenes en 59 grupos**.

**Tras dedup pHash (threshold=5):** 178 imágenes únicas, 72 duplicadas movidas a `data/raw/images/duplicates/`.

**Videos:** 2 archivos, 2.75 GB, 6m06s totales. Ambos 2160×3840 vertical, h264.
- VID_20260505_154533.mp4: 2:05 @ 60fps, 7510 frames.
- VID_20260505_155231.mp4: 4:01 @ 56fps, 13620 frames.

**Tras extract_frames con interval=15 + dedup pHash:** 1,193 frames únicos extraídos (461 + 732). Descartados por similitud: 216.

**Densidad observada en muestras visuales:** muy variable (25, 50 y 200+ alevines por imagen en las 3 muestras inspeccionadas). El modelo necesita ver TODOS estos regímenes.

---

## 3. Estimación de datos necesarios para empezar

> **Regla clave en detección de objetos: lo que importa NO es el número de imágenes, sino el número de _instancias etiquetadas_ (cajas de alevines).** Una imagen con 50 alevines aporta 50 instancias. Por eso este proyecto, aunque tenga "pocas" imágenes, puede ser viable.

### Estimación de instancias por imagen

Asumiendo que cada imagen tuya tiene en promedio **30-50 alevines visibles** (ajusta el cálculo si tus imágenes tienen más o menos):

| Imágenes etiquetadas | Instancias aproximadas | Suficiencia |
|---:|---:|---|
| 50 | ~1,500-2,500 | Muy ajustado, solo para prueba de concepto |
| **150** | **~4,500-7,500** | **Mínimo razonable para baseline de tesis** |
| **250** | **~7,500-12,500** | **Recomendado para primer entrenamiento serio** |
| 400 | ~12,000-20,000 | Sólido para reportar en tesis |
| 800+ | ~24,000+ | Óptimo, comparable a benchmarks pequeños |

### Recomendación concreta con lo que tienes

Tienes **257 imágenes originales + 5 videos de 2-3 min**.

**Plan sugerido:**

1. **Etiqueta primero 150 imágenes** del conjunto de raw (`data/raw/images/`). Elige las más diversas: distinta densidad (pocos/medianos/muchos alevines), distinta iluminación, distintos ángulos. Esto te da una **baseline funcional en ~1 semana de etiquetado**.

2. **Entrena el modelo v1** con esas 150 imágenes. Métrica objetivo de "vale la pena seguir": `mAP@50 > 0.60` y `MAE < 15%` del conteo real.

3. **Aplica pseudo-labeling** sobre las 107 imágenes restantes + frames seleccionados. El modelo v1 genera etiquetas automáticas y tú solo corriges errores. Esto te ahorra ~70% del tiempo de etiquetado.

4. **Entrena v2** con todo el dataset corregido (~250-400 imágenes). Aquí ya tienes números reportables para la tesis.

### Cantidad de imágenes vs cantidad de videos

**Las imágenes pesan más que los videos para entrenar.** Los frames de video tienden a ser similares entre sí (el mismo tanque, mismas condiciones), mientras que las imágenes pueden haber sido tomadas en momentos y ángulos distintos.

| Tipo de dato | Aporte real al modelo |
|---|---|
| 257 imágenes originales | **Alto** — cada una es una escena potencialmente única |
| 5 videos de 2-3 min | **Medio-bajo** si son del mismo tanque/condiciones; alto si son escenarios distintos |

### Sobre los videos: tiempo y cantidad

**Duración por video:**
- **30 segundos a 1 minuto** ya basta si la escena es estática (mismo tanque, mismo punto de vista, alevines moviéndose).
- **2-3 minutos** está bien si hay cambios visibles de iluminación, densidad o movimiento.
- **Más de 3 minutos** del mismo plano = redundancia. No agrega información útil.

**Cantidad de videos:**
- 5 videos está bien **si capturan condiciones distintas** entre sí:
  - Distintas densidades de alevines (tanque con pocos vs lleno).
  - Distintas iluminaciones (día/luz artificial/sombra).
  - Distintos fondos (tanque vacío al fondo vs con vegetación).
  - Distintos momentos (alimentación, descanso).
- Si los 5 videos son del mismo tanque a la misma hora, **funcionalmente cuentan como uno solo**. En ese caso convendría grabar 2-3 videos adicionales en condiciones distintas.

### Estimación esperada de frames útiles

Con tus 5 videos (asumiendo ~30 fps):
- A intervalo 30: ~900-1,500 frames totales.
- Tras deduplicación pHash (threshold 5): probablemente queden **150-400 frames únicos útiles**.
- Tras revisión manual (descartar borrosos, mal iluminados): **80-200 frames listos para etiquetar**.

Combinado con las 257 imágenes originales, el dataset etiquetable final podría llegar a **~400-450 imágenes** sin grabar más video. Ese rango es **muy bueno** para la primera versión.

### Veredicto

**Sí, ya puedes empezar.** Con 150 imágenes etiquetadas + frames seleccionados ya tienes data suficiente para producir un modelo funcional que justifique la tesis. Si los resultados del baseline no son aceptables (`mAP@50 < 0.50`), entonces sí grabar más video en condiciones distintas.

**No esperes a tener el dataset perfecto antes de entrenar.** El ciclo "etiqueto poco → entreno → veo qué falla → consigo más data del tipo correcto" es mucho más eficiente que "etiqueto todo a ciegas y entreno una vez".

---

## 4. Estrategia de etiquetado eficiente

Para no morir etiquetando manualmente cada uno de los 30-50 alevines por imagen × 250 imágenes:

1. **Usa Roboflow** (gratuito para datasets pequeños). Tiene shortcut de teclado para crear cajas rápidas y exporta directo a formato YOLO.
2. **Etiqueta en orden de diversidad, no alfabético.** Mezcla imágenes con pocos, medianos y muchos alevines desde el inicio.
3. **Sesiones de 30-45 min máximo.** Etiquetar fatiga y baja la calidad. Mejor 3 sesiones de 30 min que una de 2 horas.
4. **Después del v1: pseudo-labeling.** Generas etiquetas con el modelo, las exportas a Roboflow y solo corriges errores (mucho más rápido que dibujar desde cero).
5. **Documenta tu criterio de etiquetado.** Por ejemplo: "se etiqueta cada alevín con > 70% del cuerpo visible". Esto va en el capítulo III de la tesis.

---

## 5. Limitaciones conocidas a documentar en la tesis

Es mejor declararlas explícitamente que dejar que un jurado las encuentre:

1. **Conteo por frame sin tracking:** la v1 cuenta cada frame de video de forma independiente, por lo que el mismo alevín se cuenta varias veces. Reportar conteos solo sobre imágenes estáticas, y mencionar tracking (ByteTrack/BoT-SORT) como trabajo futuro.
2. **Oclusiones a alta densidad:** cuando los alevines se solapan, el modelo puede subcontar. Reportar el error por rango de densidad (pocos/medianos/muchos) para hacerlo transparente.
3. **Generalización a otros tanques/condiciones:** el modelo es tan bueno como la diversidad del dataset. Si solo se entrenó con imágenes de un tanque, declarar que la generalización a otros tanques requiere reentrenamiento o fine-tuning.
4. **Tamaño del dataset:** un dataset de ~250-400 imágenes es modesto. Justificar la viabilidad con: transfer learning, augmentations, alta densidad de instancias por imagen.
5. **Inferencia en CPU vs GPU:** si solo se entrena en CPU, reportar tiempos de inferencia honestos y mencionar que el despliegue móvil usará aceleración por hardware (NNAPI/CoreML/GPU del dispositivo).

---

## 6. Plan de acción inmediato (siguientes 2-3 semanas)

| Semana | Tarea | Entregable |
|---|---|---|
| 1 | Instalar venv + dependencias. Copiar imágenes y videos a `data/raw/`. Ejecutar `extract_frames.py --dedup`. Seleccionar manualmente ~50 frames útiles. | Carpeta `data/frames/selected/` con frames diversos. |
| 1-2 | Subir 150 imágenes a Roboflow y etiquetar. Exportar dataset en formato YOLOv8. | `data/dataset_yolo/` poblado. |
| 2 | Ejecutar `split_dataset.py`. Entrenar primer modelo: `python scripts/train_yolo.py --model yolov8n.pt --epochs 100`. | `best.pt` en `runs/detect/alevines_run/weights/`. |
| 2 | Validar: `yolo detect val ...`. Probar predicción en imágenes del test set con `predict_image.py`. | Métricas iniciales (P/R/mAP) en consola. |
| 3 | Llenar `data/counts/conteo_manual.csv` con ~30-50 imágenes. Ejecutar `evaluate_counting.py`. | `conteo_predicho.csv` y `reports/resultados_conteo.csv` con MAE/RMSE/MAPE. |
| 3 | Decisión: si métricas son buenas → seguir a comparación de modelos. Si no → revisar dataset, intentar `imgsz=800` o entrenar v8s. | Decisión documentada aquí. |

---

## 7. Comandos de referencia rápida

```powershell
# Activar entorno virtual (cada vez que abras una terminal nueva)
venv\Scripts\activate

# Extracción de frames recomendada
python scripts/extract_frames.py --interval 30 --dedup

# Split del dataset (cuando ya esté etiquetado)
python scripts/split_dataset.py

# Entrenamiento baseline
python scripts/train_yolo.py --model yolov8n.pt --epochs 100 --imgsz 640 --batch 8

# Validación oficial Ultralytics
yolo detect val model=runs/detect/alevines_run/weights/best.pt data=data/dataset_yolo/data.yaml

# Predicción en una imagen
python scripts/predict_image.py --weights runs/detect/alevines_run/weights/best.pt --source data/raw/images/img_001.jpg

# Evaluación de conteo manual vs automático
python scripts/evaluate_counting.py --weights runs/detect/alevines_run/weights/best.pt --images-dir data/raw/images
```

---

## 8. Criterios para considerar el modelo "aceptable para tesis"

Esto es subjetivo y depende del jurado, pero como referencia:

| Métrica | Aceptable | Bueno | Excelente |
|---|---|---|---|
| mAP@50 | > 0.60 | > 0.75 | > 0.85 |
| mAP@50-95 | > 0.35 | > 0.50 | > 0.65 |
| Precision | > 0.70 | > 0.80 | > 0.90 |
| Recall | > 0.65 | > 0.75 | > 0.85 |
| MAE de conteo | < 15% del conteo real | < 10% | < 5% |
| MAPE | < 20% | < 15% | < 10% |

Si el v1 cae en "aceptable", ya tienes algo defendible. De ahí en adelante todo es optimización.

---

*Si en futuras sesiones cambias de estrategia o decides algo distinto a lo de arriba, actualiza este archivo. Es la fuente de verdad para el proyecto.*
