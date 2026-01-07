# Proyecto CDP

Repositorio enfocado en la **fase de modelado** de un caso de **riesgo crediticio**, cuyo objetivo es predecir si un cliente es **apto o no apto para crédito** a partir de información histórica.  
El proyecto sigue una estructura tipo **MLOps pipeline**, separando datos, código, modelos y resultados para facilitar reproducibilidad, trazabilidad y escalabilidad.

---

## Objetivo del proyecto

- Construir un flujo completo de **preparación de datos → entrenamiento → evaluación → despliegue (base) → monitoreo (base)**.
- Entrenar y comparar modelos de machine learning para seleccionar el **mejor modelo** según desempeño y consistencia.
- Almacenar artefactos del pipeline (datasets procesados, modelos y métricas) de forma organizada.

---

## Estructura del repositorio

La estructura real del repositorio es la siguiente:

```text
repo_modelado_ClaraO/
└── mlops_pipeline/
    ├── data/                # Datasets procesados y artefactos de datos
    ├── models/              # Modelos entrenados y serializados
    ├── results/             # Métricas, reportes y resultados
    └── src/                 # Código fuente y notebooks
        ├── BD_creditos.xlsx
        ├── Cargar_datos.ipynb
        ├── comprension_eda.ipynb
        ├── config.json
        ├── ft_engineering.py
        ├── heuristic_model.py
        ├── model_training.py
        ├── model_evaluation.py
        ├── model_deploy.py
        └── model_monitoring.py


📦 Dataset

Ubicación: mlops_pipeline/src/BD_creditos.xlsx

El dataset contiene información histórica de clientes y su comportamiento crediticio.
Incluye variables sociodemográficas y financieras, así como la variable objetivo que indica si una persona es apta o no apta para crédito.

⸻

🔍 Análisis Exploratorio de Datos (EDA)

Archivo: mlops_pipeline/src/comprension_eda.ipynb

En esta etapa se realiza:
	•	Exploración de la estructura del dataset
	•	Identificación de valores nulos y atípicos
	•	Análisis de distribuciones de variables
	•	Exploración de la variable objetivo
	•	Definición de criterios para limpieza y transformación de datos

Los resultados del EDA guían las decisiones del feature engineering.

⸻

⚙️ Feature Engineering

Archivo: mlops_pipeline/src/ft_engineering.py

Este script se encarga de preparar los datos para el modelado:
	•	Limpieza de datos
	•	Manejo de valores nulos
	•	Codificación de variables categóricas
	•	Escalamiento de variables numéricas (cuando aplica)
	•	Generación de datasets listos para entrenamiento

Salidas esperadas:
Los datasets transformados se almacenan en mlops_pipeline/data/.

Modelo heurístico (baseline)

Archivo: mlops_pipeline/src/heuristic_model.py

Se implementa un modelo base heurístico que sirve como punto de comparación para los modelos de machine learning.
Permite validar que los modelos entrenados aportan una mejora real frente a reglas simples.

⸻

🤖 Entrenamiento y selección de modelos

Archivo: mlops_pipeline/src/model_training.py

En esta etapa se:
	•	Entrenan distintos modelos de machine learning
	•	Evalúan mediante métricas apropiadas
	•	Comparan resultados entre modelos
	•	Selecciona el mejor modelo, considerando desempeño y consistencia

Salidas esperadas:
	•	Modelo seleccionado almacenado en mlops_pipeline/models/
	•	Métricas y resultados en mlops_pipeline/results/

## 🔎 Calidad de código y análisis estático

Durante el desarrollo del proyecto se realizaron pruebas de **calidad de código** utilizando **SonarCloud**, con el objetivo de evaluar:

- Calidad y mantenibilidad del código
- Detección de code smells
- Posibles vulnerabilidades
- Buenas prácticas de desarrollo

Estas validaciones permiten asegurar que el código cumple con estándares adecuados para su integración en un entorno productivo y facilitan su escalabilidad y mantenimiento.
