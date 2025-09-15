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

<<<<<<< Updated upstream
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
=======
REM Paso 3: Crear entorno virtual si no existe
set "VENV_DIR=!project_code!-venv"
if not exist "!VENV_DIR!" (
    echo Creando entorno virtual: !VENV_DIR!
    py -m venv "!VENV_DIR!"
) else (
    echo Entorno virtual ya existe: !VENV_DIR!
)

REM Paso 4: Instalar requirements.txt si existe
call "!VENV_DIR!\Scripts\activate.bat"
if exist requirements.txt (
    echo Instalando librerías desde requirements.txt...
    pip install --no-cache-dir -r requirements.txt
) else (
    echo ADVERTENCIA: No se encontró requirements.txt, se omite instalación.
)

REM Paso 5: Asegurar ipykernel instalado
pip show ipykernel >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Instalando ipykernel...
    pip install ipykernel
)

REM Paso 6: Registrar kernel en Jupyter
python -m ipykernel install --user --name="!project_code!-venv" --display-name="Python (!project_code!-venv)"
if %ERRORLEVEL% EQU 0 (
    echo Kernel registrado correctamente como "Python (!project_code!-venv)"
>>>>>>> Stashed changes
) else (
    echo.
    echo Error activando el ambiente virtual.
)

REM Paso 7: Instrucción final para usuario
echo.
<<<<<<< Updated upstream
=======
echo === Setup finalizado ===
echo Activa tu entorno manualmente con:
echo.
echo     call "!VENV_DIR!\Scripts\activate.bat"
echo.
echo Luego abre VS Code y selecciona ese intérprete.
echo =========================

popd
>>>>>>> Stashed changes
