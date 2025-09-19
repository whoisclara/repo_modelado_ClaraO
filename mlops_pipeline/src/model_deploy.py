import time
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ================================
#     Configuración de rutas
# ================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH_BEST = BASE_DIR / "models" / "mejor_modelo.pkl"
MODEL_PATH_ALT  = BASE_DIR / "models" / "modelo_alternativo.pkl"
DATA_PATH = BASE_DIR / "data" / "df_escalado.csv"      

# ================================
#     Esperar modelos disponibles
# ================================
for path in [MODEL_PATH_BEST, MODEL_PATH_ALT]:
    print(f"🔎 Buscando modelo en: {path.resolve()}")
    while not path.exists():
        time.sleep(3)
    print(f"✅ Modelo encontrado: {path.name}")

# ================================
#     Cargar modelos y columnas
# ================================
model_best = joblib.load(MODEL_PATH_BEST)
model_alt  = joblib.load(MODEL_PATH_ALT)

df_ref = pd.read_csv(DATA_PATH)
feature_names = df_ref.drop(columns=["Pago_atiempo"]).columns.tolist()

# ================================
#        Inicializar FastAPI
# ================================
app = FastAPI(
    title="API - Modelos de Riesgo de Crédito",
    description="Sirve dos modelos de ML: uno oficial (best) y uno alternativo (alt).",
    version="2.0.0"
)

# ================================
#        Esquemas de entrada
# ================================
class InputData(BaseModel):
    features: List[float]

class BatchData(BaseModel):
    batch: List[List[float]]

# ================================
#     Endpoints modelo oficial
# ================================
@app.post("/predict_one_best")
def predict_one_best(data: InputData):
    df = pd.DataFrame([data.features], columns=feature_names)
    pred = model_best.predict(df)[0]
    proba = model_best.predict_proba(df)[0].tolist() if hasattr(model_best, "predict_proba") else []
    return {
        "modelo": "best",
        "prediction": int(pred),
        "probabilities": proba
    }

@app.post("/predict_batch_best")
def predict_batch_best(data: BatchData):
    df = pd.DataFrame(data.batch, columns=feature_names)
    preds = model_best.predict(df).tolist()
    probas = model_best.predict_proba(df).tolist() if hasattr(model_best, "predict_proba") else []
    return {
        "modelo": "best",
        "predictions": preds,
        "probabilities": probas
    }

# ================================
#     Endpoints modelo alternativo
# ================================
@app.post("/predict_one_alt")
def predict_one_alt(data: InputData):
    df = pd.DataFrame([data.features], columns=feature_names)
    pred = model_alt.predict(df)[0]
    proba = model_alt.predict_proba(df)[0].tolist() if hasattr(model_alt, "predict_proba") else []
    return {
        "modelo": "alt",
        "prediction": int(pred),
        "probabilities": proba
    }

@app.post("/predict_batch_alt")
def predict_batch_alt(data: BatchData):
    df = pd.DataFrame(data.batch, columns=feature_names)
    preds = model_alt.predict(df).tolist()
    probas = model_alt.predict_proba(df).tolist() if hasattr(model_alt, "predict_proba") else []
    return {
        "modelo": "alt",
        "predictions": preds,
        "probabilities": probas
    }

# ================================
#     Endpoint raíz
# ================================
@app.get("/")
def read_root():
    return {
        "mensaje": "✅ API activa. Visita /docs para ver los endpoints disponibles.",
        "endpoints": [
            "/predict_one_best", "/predict_batch_best",
            "/predict_one_alt", "/predict_batch_alt"
        ]
    }
