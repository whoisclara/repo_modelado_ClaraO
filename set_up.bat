@echo off
setlocal EnableDelayedExpansion EnableExtensions

echo === Python Virtual Environment Setup ===
echo.

REM Paso 1: Navegar a carpeta donde está config.json
pushd "%~dp0mlops_pipeline\src" || (
    echo ERROR: No se encontró la carpeta "%~dp0mlops_pipeline\src"
    echo Verifica la ruta o el nombre de carpeta.
    exit /b 1
)

REM Paso 2: Leer project_code de config.json
for /f "usebackq tokens=2 delims=:" %%A in (`findstr "project_code" config.json`) do (
    set "line=%%A"
    set "line=!line:,=!"
    set "line=!line:"=!"
    set "project_code=!line:~1!"
)

echo Proyecto detectado: !project_code!

REM Volver al root del repo
popd
pushd "%~dp0"

REM Paso 3: Crear entorno virtual
set "VENV_DIR=!project_code!-venv"
echo Creando entorno virtual: !VENV_DIR!
py -m venv "!VENV_DIR!"

REM Paso 4: Activar entorno virtual
call "!VENV_DIR!\Scripts\activate.bat" || (
    echo ERROR: No se pudo activar el entorno virtual.
    exit /b 1
)

echo Entorno virtual activo: %VIRTUAL_ENV%
where python

REM Paso 5: Instalar requirements.txt si existe
if exist requirements.txt (
    echo Instalando librerías desde requirements.txt...
    pip install --no-cache-dir -r requirements.txt
) else (
    echo ADVERTENCIA: No se encontró requirements.txt, se omite instalación.
)

REM Paso 6: Verificar ipykernel
echo Verificando si ipykernel está instalado...
pip show ipykernel >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ipykernel no encontrado, instalando...
    pip install ipykernel
)

REM Paso 7: Registrar kernel en Jupyter
echo Registrando kernel Jupyter...
python -m ipykernel install --user --name="!project_code!-venv" --display-name="Python (.venv) - MLOps"
if %ERRORLEVEL% EQU 0 (
    echo Kernel registrado correctamente como "Python (.venv) - MLOps"
) else (
    echo ERROR: Falló el registro del kernel.
)

popd
echo.
echo Setup finalizado. Puedes iniciar JupyterLab ahora.
