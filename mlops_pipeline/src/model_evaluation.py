import requests
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
API_URL = "http://127.0.0.1:8000/predict_batch"

st.title("📊 Evaluación del Modelo Desplegado")


st.write("Cargando dataset de test...")

try:
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
except FileNotFoundError:
    X_test = pd.read_csv(DATA_DIR / "X_test_raw.csv")

y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

st.write("Dataset de test cargado con éxito")
st.write(f"Tamaño X_test: {X_test.shape}, y_test: {y_test.shape}")

#LLamar al end-point
st.write("Obteniendo predicciones del modelo vía API...")

payload = {"batch": X_test.values.tolist()}
response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    result = response.json()
    preds = result["predictions"]
    probas = result.get("probabilities", None)
else:
    st.error(f"Error al conectar con la API ({response.status_code})")
    st.stop()

# ============================
# Métricas
# ============================
st.subheader("Métricas de Clasificación")
report = classification_report(y_test, preds, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose()

st.dataframe(df_report.style.background_gradient(cmap="Blues").format("{:.2f}"))

# ============================
# Matriz de confusión
# ============================
st.subheader("Matriz de Confusión")
cm = confusion_matrix(y_test, preds)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0,1], yticklabels=[0,1], ax=ax)
ax.set_xlabel("Predicciones")
ax.set_ylabel("Reales")
st.pyplot(fig)

# ============================
# Curva ROC
# ============================
st.subheader("Curva ROC")
if probas:
    probs_clase1 = [p[1] for p in probas]
    fpr, tpr, _ = roc_curve(y_test, probs_clase1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1], [0,1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    st.pyplot(fig)
else:
    st.warning("El modelo no retornó probabilidades, no se puede calcular ROC.")

# ============================
#  Precision-Recall Curve
# ============================
st.subheader("Curva Precision-Recall")

if probas:
    probs_clase1 = [p[1] for p in probas]  # probabilidad clase positiva
    precision, recall, thresholds = precision_recall_curve(y_test, probs_clase1)
    ap_score = average_precision_score(y_test, probs_clase1)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AP = {ap_score:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.legend(loc="lower left")
    st.pyplot(fig)
else:
    st.warning("El modelo no retornó probabilidades, no se puede calcular Precision-Recall.")
