# mlops_pipeline/src/model_deploy.py

import time
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
# ================================
#     Configuración de rutas
# ================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "mejor_modelo.pkl"  # Asegúrate de que exista
DATA_PATH = BASE_DIR / "data" / "df_escalado.csv"      # Dataset de referencia

# ================================
#     Esperar a que el modelo esté disponible
# ================================

print("🔎 Buscando modelo en:", MODEL_PATH.resolve())
print("⏳ Esperando a que el modelo esté disponible...")
while not MODEL_PATH.exists():
    time.sleep(3)
print("✅ Modelo encontrado. Cargando...")

# ================================
#     Cargar modelo y columnas
# ================================

model = joblib.load(MODEL_PATH)
df_ref = pd.read_csv(DATA_PATH)
feature_names = df_ref.drop(columns=["Pago_atiempo"]).columns.tolist()

# ================================
#        Inicializar FastAPI
# ================================

app = FastAPI(
    title="API - Modelo de Riesgo de Crédito",
    description="Esta API sirve un modelo de ML para predecir si un cliente pagará a tiempo. Usa /predict_one para una sola instancia o /predict_batch para predicciones por lote.",
    version="1.0.0"
)

# ================================
#        Esquemas de entrada
# ================================

class InputData(BaseModel):
    features: List[float]

class BatchData(BaseModel):
    batch: List[List[float]]

# ================================
#     Endpoint para un registro
# ================================

@app.post("/predict_one")
def predict_one(data: InputData):
    df = pd.DataFrame([data.features], columns=feature_names)
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0].tolist() if hasattr(model, "predict_proba") else []
    return {
        "prediction": int(pred),
        "probabilities": proba
    }

# ================================
#     Endpoint para batch
# ================================

@app.post("/predict_batch")
def predict_batch(data: BatchData):
    df = pd.DataFrame(data.batch, columns=feature_names)
    preds = model.predict(df).tolist()
    probas = model.predict_proba(df).tolist() if hasattr(model, "predict_proba") else []
    return {
        "predictions": preds,
        "probabilities": probas
    }

# ================================
#     Endpoint raíz (evita 404)
# ================================

@app.get("/")
def read_root():
    return {
        "mensaje": "✅ API activa. Visita /docs para ver los endpoints disponibles.",
        "endpoints": ["/predict_one", "/predict_batch"]
    }

