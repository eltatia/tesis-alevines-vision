"""
prepare_collab_setup.py
-----------------------
Agrega a cada lote_N de data/labeling_collab/ todo lo necesario
para que el colaborador pueda etiquetar sin pelearse con bugs de PyQt5:

  - _parches/  con los 3 archivos parcheados (canvas.py, shape.py, labelImg.py)
  - setup.bat  crea venv local, instala labelImg, aplica parches
  - launch_labelimg.bat  actualizado para usar venv local
  - INSTRUCCIONES.txt  actualizado al nuevo flow

Uso:
    python scripts/prepare_collab_setup.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SETUP_BAT = r"""@echo off
REM Setup automatico para etiquetado colaborativo.
REM 1) Crea venv local
REM 2) Instala labelImg
REM 3) Aplica parches de PyQt5 (Python 3.10+)
REM 4) Limpia cache

cd /d "%~dp0"

echo === 1) Verificando Python ===
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python no encontrado. Instala Python 3.10 desde:
    echo   https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version

echo.
echo === 2) Creando entorno virtual local (venv\) ===
if not exist venv (
    python -m venv venv
)

echo.
echo === 3) Instalando labelImg ===
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install labelImg

echo.
echo === 4) Aplicando parches de PyQt5 ===
copy /Y "_parches\canvas.py"   "venv\Lib\site-packages\libs\canvas.py"   >nul
copy /Y "_parches\shape.py"    "venv\Lib\site-packages\libs\shape.py"    >nul
copy /Y "_parches\labelImg.py" "venv\Lib\site-packages\labelImg\labelImg.py" >nul

echo.
echo === 5) Limpiando cache de Python ===
if exist "venv\Lib\site-packages\libs\__pycache__" (
    del /Q "venv\Lib\site-packages\libs\__pycache__\*.pyc" >nul 2>nul
)
if exist "venv\Lib\site-packages\labelImg\__pycache__" (
    del /Q "venv\Lib\site-packages\labelImg\__pycache__\*.pyc" >nul 2>nul
)

echo.
echo === Setup completado ===
echo Ahora podes ejecutar launch_labelimg.bat con doble click.
pause
"""


LAUNCH_BAT = r"""@echo off
REM Abre LabelImg con esta carpeta ya cargada (usa venv local parcheado)
cd /d "%~dp0"

REM %~dp0 termina en "\" y eso escapa la comilla siguiente, rompiendo los
REM argumentos. Le quitamos la barra final antes de usarlo.
set "LOTE=%~dp0"
if "%LOTE:~-1%"=="\" set "LOTE=%LOTE:~0,-1%"

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] No existe el entorno local.
    echo Ejecuta primero setup.bat en esta misma carpeta.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Abriendo LabelImg...
echo   Ctrl + rueda del mouse = zoom
echo   Boton central (rueda) arrastrando = mover la imagen
echo.
labelImg "%LOTE%" "%LOTE%\classes.txt" "%LOTE%"

if errorlevel 1 (
    echo.
    echo ============================================================
    echo [ERROR] LabelImg se cerro con error. El mensaje esta arriba.
    echo Mandale una captura de esta ventana al responsable.
    echo ============================================================
    pause
)
"""


INSTRUCCIONES_TEMPLATE = """\
INSTRUCCIONES PARA ETIQUETADO COLABORATIVO
==========================================

Este lote contiene {n_imagenes} imagenes de alevines con etiquetas
pre-generadas automaticamente (cajas verdes ya dibujadas).

Tu tarea: revisar cada imagen y CORREGIR las etiquetas.

PASOS (solo la primera vez):
----------------------------
1. Instalar Python 3.10 si no lo tenes:
   https://www.python.org/downloads/
   IMPORTANTE: marcar la casilla "Add Python to PATH" al instalar.

2. Doble click en setup.bat
   (instala labelImg en un venv local y aplica los parches necesarios)

PASOS (cada vez que vayas a etiquetar):
---------------------------------------
3. Doble click en launch_labelimg.bat

4. En LabelImg:
   - Cambiar formato a YOLO: clickear el boton de la sidebar izquierda
     hasta que diga "YOLO" (por defecto arranca en "PascalVOC")
   - Activar Auto Save: menu View -> Auto Save mode

5. Atajos esenciales:
   W = nueva caja        |  D = siguiente imagen
   Ctrl+S = guardar      |  A = imagen anterior
   Del = borrar caja     |  Ctrl+D = duplicar caja
   Rueda del mouse = scroll | Ctrl+rueda = zoom

6. Para cada imagen:
   - Borrar cajas falsas (zapatos, marca de bandeja, sombras del agua)
   - Agregar alevines que no estan marcados (W para dibujar nueva caja)
   - Ajustar cajas mal posicionadas (arrastrar las esquinas)
   - Guardar antes de pasar a la siguiente (Auto Save lo hace solo)

CRITERIO DE ETIQUETADO:
-----------------------
- Etiquetar cada alevin visible con mas del 50% del cuerpo en cuadro
- NO etiquetar alevines totalmente ocluidos por otros
- UNA caja por alevin (no por cabeza+cola por separado)
- En cumulos densos, etiquetar solo los que se distinguen claramente
- En las imagenes muy densas (con 100+ alevines amontonados) hacer lo mejor
  posible pero NO obsesionarse - documentaremos como limitacion.

DEVOLUCION:
-----------
Cuando termines (o al hacer pausa para sincronizar):
- Comprimir TODA esta carpeta (excepto venv\\ que pesa mucho) en un ZIP.
- O mejor: comprimir solo los archivos .jpg y .txt al mismo nivel que esta
  carpeta. No necesitas devolver venv\\ ni _parches\\
- Enviarlo de vuelta por Drive/WeTransfer.
- El responsable principal lo procesara con scripts/merge_collab.py

NOTA TECNICA:
-------------
La carpeta venv\\ son las dependencias de Python (~600 MB), NO tocar.
La carpeta _parches\\ tiene los archivos para arreglar bugs de la GUI.
classes.txt define las clases (en este proyecto solo "alevin").

Si tenes algun problema, contactar al responsable principal.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--collab-dir", type=str, default="data/labeling_collab")
    p.add_argument("--venv-libs", type=str,
                   default="venv/Lib/site-packages",
                   help="Ruta al site-packages del venv principal (origen de parches)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    collab_dir = Path(args.collab_dir)
    venv_libs = Path(args.venv_libs)

    if not collab_dir.exists():
        raise SystemExit(f"[ERROR] No existe {collab_dir}. Ejecuta primero split_for_collab.py")

    canvas_src = venv_libs / "libs" / "canvas.py"
    shape_src = venv_libs / "libs" / "shape.py"
    labelimg_src = venv_libs / "labelImg" / "labelImg.py"

    for f in [canvas_src, shape_src, labelimg_src]:
        if not f.exists():
            raise SystemExit(f"[ERROR] No se encuentra archivo parcheado: {f}")

    lotes = sorted([p for p in collab_dir.iterdir() if p.is_dir() and p.name.startswith("lote_")])
    if not lotes:
        raise SystemExit(f"[ERROR] No hay subcarpetas lote_N en {collab_dir}")

    for lote in lotes:
        # Crear _parches/
        parches_dir = lote / "_parches"
        parches_dir.mkdir(exist_ok=True)
        shutil.copy2(str(canvas_src), str(parches_dir / "canvas.py"))
        shutil.copy2(str(shape_src), str(parches_dir / "shape.py"))
        shutil.copy2(str(labelimg_src), str(parches_dir / "labelImg.py"))

        # Generar setup.bat y launch_labelimg.bat
        (lote / "setup.bat").write_text(SETUP_BAT, encoding="utf-8")
        (lote / "launch_labelimg.bat").write_text(LAUNCH_BAT, encoding="utf-8")

        # Contar imagenes para INSTRUCCIONES
        n_imagenes = len([p for p in lote.iterdir()
                          if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        (lote / "INSTRUCCIONES.txt").write_text(
            INSTRUCCIONES_TEMPLATE.format(n_imagenes=n_imagenes), encoding="utf-8"
        )

        print(f"  {lote.name}: configurado ({n_imagenes} imagenes)")

    print(f"\nListo. Cada lote ahora incluye:")
    print(f"  _parches/             3 archivos para arreglar bugs PyQt5")
    print(f"  setup.bat             instala labelImg + aplica parches")
    print(f"  launch_labelimg.bat   abre labelImg con todo configurado")
    print(f"  INSTRUCCIONES.txt     guia para el colaborador")
    print(f"\nProximo paso: comprimir cada lote_N en ZIP y enviar por Drive.")


if __name__ == "__main__":
    main()
