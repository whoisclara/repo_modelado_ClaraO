# Proyecto CDP

Este proyecto se desarrolla con el propósito de construir un pipeline de ciencia de datos orientado a la predicción de la aptitud crediticia de una persona, a partir de información histórica. La pregunta central que guía el trabajo es si, dados ciertos datos disponibles, es posible estimar de manera confiable si un individuo es apto o no para acceder a un crédito.

A lo largo del proyecto se plantea un flujo organizado que permite trabajar los datos de forma estructurada, comenzando por su análisis y comprensión, continuando con su transformación y preparación, y culminando en el entrenamiento y evaluación de modelos predictivos. Más allá de obtener un resultado puntual, el enfoque está puesto en la construcción de un proceso claro y reproducible, que refleje cómo este tipo de soluciones podrían desarrollarse en un entorno real.


## Objetivo del proyecto

Predecir si un cliente pagará su crédito a tiempo utilizando técnicas avanzadas de machine learning y un pipeline de producción robusto:

Clase 1: Cliente paga a tiempo (Pago_atiempo = 1)
Clase 0: Cliente moroso (Pago_atiempo = 0)

Meta de negocio: Detectar al menos 75% de los clientes morosos para minimizar pérdidas financieras.

---

## Estructura del repositorio

La estructura real del repositorio es la siguiente:

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

## Dataset

El dataset contiene información histórica de clientes y su comportamiento crediticio.
Incluye variables sociodemográficas y financieras, así como la variable objetivo que indica si una persona es apta o no apta para crédito.

## Features Implementadas
**Developer**: Ingeniería de Datos
Limpieza y preprocesamiento robusto
Feature selection basada en domain knowledge
Transformaciones optimizadas por tipo de modelo

**Feature 1**: Entrenamiento y Despliegue
Ensemble de modelos extremos con class weights
Optimización de hiperparámetros automática
API REST containerizada lista para producción

**Feature 2**: Monitoreo y Evaluación
Dashboard real-time de data drift
Sistema de alertas automático
Métricas de negocio alineadas con objetivos

## Calidad de código y análisis estático

Durante el desarrollo del proyecto se realizaron pruebas de calidad de código utilizando SonarCloud, con el objetivo de evaluar:

- Calidad y mantenibilidad del código
- Detección de code smells
- Posibles vulnerabilidades
- Buenas prácticas de desarrollo

## Conclusiones

Este proyecto demuestra la implementación de un pipeline MLOps completo para scoring de crédito que incluye:

- Ingeniería de datos robusta con eliminación de data leakage
- Modelos ensemble extremos optimizados para detectar morosos
- API de producción con FastAPI y containerización Docker
- Monitoreo automático de data drift con alertas inteligentes
- Métricas de negocio alineadas con objetivos financieros reales

Por si solo no se recomienda usar un único modelo para hacer la predicción necesaria para este problema, pues ningún modelo es bueno prediciendo si una persona es morosa o no, como se puedo evidenciar en las métricas de evaluación, ya que predice demasiado bien una variable o la otra pero no las dos al tiempo. Por lo que se recomienda integrar en un futuro proyecto dos modelos de machine learning que funcionen muy bien para cada una de las respuestas del target, para así entrenar un nuevo modelo capaz de resolverlo.

## Licencia

Proyecto académico para la materia Ciencia de Datos en Producción - Universidad Pontificia Bolivariana.

## Autor

ClaraIsabel Otalvaro Agudelo
e-mail: clara.otalvaro@upb.edu.co
Fecha: 25 de septiembre
