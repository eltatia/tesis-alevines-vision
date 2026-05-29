@echo off
REM Abre LabelImg con data/labeling_subset/ (114 imagenes estratificadas)
REM con sus etiquetas auto-generadas para revisar/corregir.

cd /d "%~dp0"
call venv\Scripts\activate.bat
labelImg data\labeling_subset data\labeling_subset\classes.txt data\labeling_subset
