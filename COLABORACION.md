# Guía para colaborador — etiquetado de alevines

Bienvenido. Este documento es para que puedas ayudar con el etiquetado del dataset de alevines del proyecto de tesis.

**Tu tarea:** revisar 58 imágenes con etiquetas pre-generadas (cajas verdes ya dibujadas) y corregirlas. **NO** vas a dibujar desde cero — solo borrar/agregar/ajustar lo que esté mal.

---

## 1. Requisitos previos

Antes de empezar necesitas tener instalado:

- **Windows 10 u 11**
- **Python 3.10** → https://www.python.org/downloads/release/python-31011/
  - ⚠️ **MUY IMPORTANTE:** al instalar marca la casilla **"Add Python to PATH"** en la primera pantalla del instalador.
- **Git** → https://git-scm.com/download/win
  - Usa la configuración por defecto en el instalador.

Para verificar que se instalaron, abre PowerShell y ejecuta:

```powershell
python --version
git --version
```

Ambos comandos deben mostrarte una versión, no un error de "no encontrado".

---

## 2. Clonar el repositorio

Abre PowerShell o CMD donde quieras tener el proyecto (ej. en tu escritorio) y ejecuta:

```powershell
git clone https://github.com/eltatia/tesis-alevines-vision.git
cd tesis-alevines-vision
```

Eso te baja **todo lo necesario**: el código, las etiquetas YOLO (.txt) **y las 58 imágenes .jpg** del lote que vas a revisar. Ya **no** necesitas descargar nada desde Google Drive.

Para confirmar que las imágenes llegaron, verifica que esta carpeta tenga archivos `.jpg`:

```
tesis-alevines-vision\data\labeling_collab\lote_companero\
```

Debe verse así:

```
tesis-alevines-vision/
└── data/
    └── labeling_collab/
        └── lote_companero/
            ├── *.jpg              ← (58 archivos) ya vienen con el git clone
            ├── *.txt              ← (58 archivos) etiquetas pre-generadas
            ├── classes.txt
            ├── INSTRUCCIONES.txt
            ├── setup.bat          ← lo vas a usar abajo
            ├── launch_labelimg.bat ← lo vas a usar abajo
            └── _parches/          ← arregla bugs de labelImg
```

---

## 4. Instalar LabelImg (solo la primera vez)

Entra a la carpeta `lote_companero` y haz **doble click** en `setup.bat`.

El script automáticamente:

1. Crea un entorno virtual local llamado `venv\` (no toca tu Python global).
2. Instala labelImg + sus dependencias (~70 MB de descarga, tarda ~5 min).
3. Aplica los parches para evitar 3 bugs conocidos de PyQt5 con Python 3.10+ (sin estos parches, LabelImg se cierra al dibujar cajas o usar la rueda del mouse).
4. Te pide que presiones una tecla cuando termina.

Cuando veas el mensaje "Setup completado" ya está listo.

---

## 5. Abrir LabelImg y etiquetar

Doble click en `launch_labelimg.bat`.

Se va a abrir la app de LabelImg con las 58 imágenes ya cargadas.

### Pasos críticos en LabelImg (la primera vez)

1. **Cambia el formato a YOLO**:
   - En la **sidebar izquierda** (columna de iconos), busca un botón que diga **"PascalVOC"**.
   - Click hasta que cambie a **"YOLO"**. Sin esto, el programa guarda en otro formato y no nos sirve.
   - Cómo confirmar que está en YOLO: cuando navegues a la primera imagen, verás cajas verdes pre-dibujadas. Si no ves cajas, el formato no es YOLO.

2. **Activa Auto Save**:
   - Menú **View** → click en **Auto Save mode** (queda con un check).
   - Esto guarda automáticamente cada vez que cambias de imagen.

### Atajos de teclado esenciales

| Tecla | Acción |
|---|---|
| `W` | Dibujar nueva caja |
| `D` | Siguiente imagen |
| `A` | Imagen anterior |
| `Ctrl+S` | Guardar (por las dudas, aunque tengas Auto Save) |
| `Del` | Borrar caja seleccionada |
| `Ctrl+D` | Duplicar caja seleccionada |
| `Rueda del mouse` | Scroll vertical |
| `Ctrl + Rueda` | Zoom in / out |

---

## 6. Qué corregir en cada imagen

Las cajas verdes pre-dibujadas vienen de un modelo de IA (FastSAM). NO son perfectas. Tu trabajo es **corregir**, específicamente:

### Borrar cajas falsas (falsos positivos)

- Cajas sobre el **zapato** del operador (al pie de algunas fotos).
- Cajas sobre la **marca de fábrica** del recipiente (suele ser un relieve en el centro).
- Cajas sobre **sombras**, reflejos del agua, manchas del fondo.
- Cualquier caja que claramente no esté sobre un alevín.

### Agregar cajas que falten (falsos negativos)

- Sobre todo en zonas densas: el modelo se pierde alevines cuando están juntos o medio ocluidos.
- Presiona `W`, dibuja la caja arrastrando con el mouse y al soltar te aparece un diálogo. Selecciona `alevin` y OK.

### Ajustar cajas mal posicionadas

- Click sobre la caja para seleccionarla.
- Arrastra las esquinas para redimensionar.
- Arrastra el centro para mover.

---

## 7. Criterios de etiquetado (importante para la tesis)

- **Etiqueta cada alevín** con más del **50% del cuerpo** visible en el cuadro.
- **NO etiquetes** alevines totalmente ocluidos por otros (los que no se distinguen).
- **UNA caja por alevín completo** (NO una para la cabeza y otra para la cola del mismo pez).
- En **cúmulos muy densos** (esquinas con muchos alevines amontonados): etiqueta solo los que se distinguen claramente, NO obsesiones con los que están en cúmulos. Vamos a documentarlo como limitación.

---

## 8. Subir tu progreso al repo (cada cierto tiempo)

Para no perder lo que hiciste y que el responsable pueda ver tus avances:

```powershell
cd ruta\donde\clonaste\tesis-alevines-vision

# Ver qué cambió
git status

# Agregar los .txt modificados
git add data/labeling_collab/lote_companero/*.txt

# Hacer commit
git commit -m "Revisar 15 imagenes del lote_companero"

# Subir al repo
git push
```

**Cuándo hacer push:**
- Cuando termines una sesión (aunque sean pocas imágenes).
- O cada 20-30 imágenes corregidas, como respaldo.
- Definitivamente cuando termines TODAS las 58.

**Si es la primera vez que usas git en esta máquina** y te pide autenticación, sigue las instrucciones que aparezcan en pantalla (te va a abrir el navegador para loguearte en GitHub).

---

## 9. Problemas comunes

### "LabelImg se cierra cuando dibujo una caja"
El parche no se aplicó. Volvé a ejecutar `setup.bat`.

### "Veo las imágenes pero sin cajas verdes"
Está en formato PascalVOC. Cambia a YOLO con el botón de la sidebar.

### "Quiero pausar y volver mañana"
Cierra LabelImg normalmente. Tus cambios están guardados (Auto Save). Antes de cerrar, hace `git push` para tener un respaldo. Cuando vuelvas, `git pull` para asegurarte de tener lo último y luego doble click en `launch_labelimg.bat`.

### "Se me trabó git"
Avisame, no fuerces nada destructivo.

### "Hay alevines que no sé si etiquetar o no"
Ante la duda, **etiquetalos**. Es más fácil borrar después que detectar lo que falta. Pero si está claramente ocluido o muy borroso, déjalo.

---

## 10. Cuando termines TODAS las 58 imágenes

Avísale al responsable principal con un mensaje:

> "Terminé las 58 imágenes del lote_companero. Ya pusheé al repo."

Él va a:
1. Hacer `git pull` en su PC.
2. Ejecutar un script que une tus correcciones al subset principal.
3. Entrenar el primer modelo con las 114 imágenes completamente revisadas.

---

## Tiempo estimado

- **Setup inicial (1 sola vez):** 15-20 minutos.
- **Etiquetar 58 imágenes:** 5-8 horas (depende de la densidad de cada imagen y de tu velocidad).
- **No tienes que hacerlo todo de una vez.** Podes ir de a poco en varias sesiones.

---

## Contacto

Si algo no funciona o tienes dudas, avísale al responsable principal. **No borres archivos del repo sin consultar**.

¡Gracias por la ayuda!
