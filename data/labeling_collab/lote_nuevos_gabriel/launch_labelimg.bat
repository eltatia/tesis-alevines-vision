@echo off
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
