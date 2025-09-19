# Imagen base de Python
FROM python:3.11-slim

# Definir directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto al contenedor
COPY requirements.txt .
COPY mlops_pipeline ./mlops_pipeline

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto donde correrá Uvicorn
EXPOSE 8000

# Comando para correr la API al iniciar el contenedor
CMD ["uvicorn", "mlops_pipeline.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
