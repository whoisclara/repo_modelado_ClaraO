# mlops_pipeline/src/model_evaluation.py
import requests
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 1. Configuración
# ============================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "df_escalado.csv"
API_URL = "http://127.0.0.1:8000/predict_batch"
TARGET = "Pago_atiempo"

st.title("📊 Evaluación del Modelo Desplegado")

# ============================
# 2. Cargar datos
# ============================
st.write("Cargando dataset escalado...")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split igual que en ft_engineering
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.write("✅ Dataset cargado con éxito")
st.write(f"Tamaño X_test: {X_test.shape}, y_test: {y_test.shape}")

# ============================
# 3. Llamar al endpoint
# ============================
st.write("Obteniendo predicciones del modelo vía API...")

payload = {"batch": X_test.values.tolist()}
response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    result = response.json()
    preds = result["predictions"]
    probas = result.get("probabilities", None)
else:
    st.error(f"❌ Error al conectar con la API ({response.status_code})")
    st.stop()

# ============================
# 4. Métricas
# ============================
st.subheader("Métricas de Clasificación")
report = classification_report(y_test, preds, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose()

# Mostrar como tabla
st.dataframe(df_report.style.background_gradient(cmap="Blues").format("{:.2f}"))

# ============================
# 5. Matriz de confusión
# ============================
st.subheader("Matriz de Confusión")
cm = confusion_matrix(y_test, preds)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1], ax=ax)
ax.set_xlabel("Predicciones")
ax.set_ylabel("Reales")
st.pyplot(fig)

# ============================
# 6. Curva ROC
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
# 7. Precision-Recall Curve
# ============================
from sklearn.metrics import precision_recall_curve, average_precision_score

st.subheader("Curva Precision-Recall")

if probas:
    probs_clase1 = [p[1] for p in probas]  # probabilidad de clase positiva
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
