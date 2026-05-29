@echo off
REM Abre LabelImg con esta carpeta ya cargada (usa venv local parcheado)
cd /d "%~dp0"
call venv\Scripts\activate.bat
labelImg "%~dp0" "%~dp0\classes.txt" "%~dp0"
