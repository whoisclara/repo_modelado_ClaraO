@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo === Python Virtual Environment Setup ===
echo.

REM Desactivar el ambiente virtual actual si está activo
if defined VIRTUAL_ENV (
    echo Desactivando ambiente virtual actual: %VIRTUAL_ENV%
    call deactivate
)

echo Buscando código del proyecto en config.json...

REM Ir a la carpeta donde está config.json (relativa al .bat)
pushd "%~dp0mlops_pipeline\src" || (
    echo ❌ No existe: "%~dp0mlops_pipeline\src"
    echo Verifica la ruta o el nombre de carpeta y vuelve a ejecutar.
    exit /b 1
)

REM Leer "project_code" de config.json
for /f "usebackq tokens=2 delims=:" %%A in (`findstr "project_code" config.json`) do (
    set "line=%%A"
    set "line=!line:,=!"
    set "line=!line:"=!"
    set "project_code=!line:~1!"
)

REM Línea de debug (opcional)
echo project_code=!project_code!

REM Volver al raíz del repo (dos niveles arriba del src)
popd
pushd "%~dp0"

echo Creando nuevo ambiente virtual: !project_code!-venv
py -m venv "!project_code!-venv"

echo Activating virtual environment...
call "!project_code!-venv\Scripts\activate"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Ambiente virtual creado con exito!.
    echo Python actual:
    where python

    echo.
    echo === Instalando requisitos ===
    if exist requirements.txt (
        echo requirements.txt encontrado, instalando librerias...
        pip install --no-cache-dir -r requirements.txt

        if %ERRORLEVEL% EQU 0 (
            echo.
            echo Todas las librerías instaladas correctamente.
            echo.
            echo === Registrando ambiente virtual con Jupyter ===
            python -m ipykernel install --user --name="!project_code!-venv" --display-name="!project_code!-venv Python ETL"
            if %ERRORLEVEL% EQU 0 (
                echo Kernel registrado: !project_code!-venv
            ) else (
                echo Advertencia: Fallo al registrar el kernel de Jupyter.
            )
        ) else (
            echo.
            echo Error instalando librerías desde requirements.txt.
        )
    ) else (
        echo.
        echo Advertencia: requirements.txt no fue encontrado en el directorio del repo.
    )
) else (
    echo.
    echo Error activando el ambiente virtual.
)

popd
echo.
