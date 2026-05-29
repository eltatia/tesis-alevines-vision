@echo off
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
