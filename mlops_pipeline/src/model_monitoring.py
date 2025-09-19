import os
import time
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from pathlib import Path

# ============================
# 1. Configuración de rutas
# ============================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "df_escalado.csv"
TARGET = "Pago_atiempo"

API_URL_BEST = "http://127.0.0.1:8000/predict_batch_best"
API_URL_ALT  = "http://127.0.0.1:8000/predict_batch_alt"

MONITOR_FILE_BEST = DATA_DIR / "monitoring_log_best.csv"
MONITOR_FILE_ALT  = DATA_DIR / "monitoring_log_alt.csv"

# ============================
# 2. Cargar dataset referencia
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y

X_ref, y_ref = load_data()

# ============================
# 3. Funciones auxiliares
# ============================
def get_predictions(X_batch: pd.DataFrame, api_url: str):
    payload = {"batch": X_batch.values.tolist()}
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        preds = response.json()["predictions"]
        return preds
    except Exception as e:
        st.error(f"❌ Error conectando con la API: {e}")
        return None

def log_predictions(X_batch, y_true, preds, log_file):
    log_df = X_batch.copy()
    log_df["y_true"] = y_true.values
    log_df["y_pred"] = preds
    log_df["timestamp"] = pd.Timestamp.now()

    if log_file.exists():
        log_df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)

def generate_drift_report(ref_data, new_data):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data, current_data=new_data)
    return report

# ============================
# 4. Función de dashboard
# ============================
def dashboard_model(nombre, api_url, log_file):
    st.header(f"📊 Dashboard - {nombre}")

    sample_size = st.sidebar.slider(f"Tamaño lote ({nombre})", 50, 500, 200, key=nombre)

    if st.sidebar.button(f"🔄 Generar nuevas predicciones ({nombre})"):
        sample = X_ref.sample(n=sample_size, random_state=int(time.time()))
        preds = get_predictions(sample, api_url)
        if preds:
            y_sample = y_ref.loc[sample.index]
            log_predictions(sample, y_sample, preds, log_file)
            st.success(f"✅ Nuevas predicciones agregadas al log de {nombre}.")
            st.rerun()

    if log_file.exists():
        df_log = pd.read_csv(log_file)

        # KPIs arriba
        col1, col2, col3, col4, col5 = st.columns(5)
        acc = accuracy_score(df_log["y_true"], df_log["y_pred"])
        prec = precision_score(df_log["y_true"], df_log["y_pred"], zero_division=0)
        rec = recall_score(df_log["y_true"], df_log["y_pred"], zero_division=0)
        pos_rate = df_log["y_pred"].mean() * 100

        with col1: st.metric("Total Registros", len(df_log))
        with col2: st.metric("Accuracy", f"{acc:.2f}")
        with col3: st.metric("Precision", f"{prec:.2f}")
        with col4: st.metric("Recall", f"{rec:.2f}")
        with col5: st.metric("Tasa Positiva", f"{pos_rate:.1f}%")

        # Tabs internas
        tab1, tab2, tab3 = st.tabs(["📈 Gráficas", "📊 Data Drift", "📂 Logs"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(df_log, x="y_pred", nbins=20, title="Distribución de Predicciones")
                st.plotly_chart(fig_hist, width="stretch")
            with col2:
                df_log["timestamp"] = pd.to_datetime(df_log["timestamp"])
                temporal = df_log.groupby(df_log["timestamp"].dt.floor("min"))["y_pred"].mean().reset_index()
                fig_time = px.line(temporal, x="timestamp", y="y_pred", title="Evolución Temporal")
                st.plotly_chart(fig_time, width="stretch")

            # Matriz de confusión
            st.subheader("Matriz de Confusión")
            cm = confusion_matrix(df_log["y_true"], df_log["y_pred"])
            z = cm.tolist()
            x = ["Pred 0", "Pred 1"]
            y = ["Real 0", "Real 1"]
            fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale="Blues", showscale=True)
            fig_cm.update_layout(title="Matriz de Confusión")
            st.plotly_chart(fig_cm, width="stretch")

            # Distribución de probabilidades (si existe en log)
            if "prob_1" in df_log.columns:
                fig_prob = px.histogram(df_log, x="prob_1", nbins=20, title="Distribución de Probabilidades (clase 1)")
                st.plotly_chart(fig_prob, width="stretch")

        with tab2:
            st.subheader("📊 Reporte de Drift con Evidently")
            drift_report = generate_drift_report(
                X_ref, df_log.drop(columns=["y_true", "y_pred", "timestamp"], errors="ignore")
            )
            try:
                st.components.v1.html(drift_report._repr_html_(), height=800, scrolling=True)
            except:
                st.write("✅ Reporte de drift generado correctamente.")

        with tab3:
            st.subheader("📂 Log de Monitoreo")
            st.dataframe(df_log.tail(50), width="stretch")
            st.download_button(
                f"📥 Descargar log completo ({nombre})",
                df_log.to_csv(index=False),
                file_name=f"monitoring_log_{nombre}.csv",
                mime="text/csv"
            )
    else:
        st.warning(f"⚠️ No hay registros aún para {nombre}.")

# ============================
# 5. UI principal
# ============================
st.set_page_config(page_title="📊 Model Monitoring", layout="wide")
st.title("📊 Dashboard Comparativo de Modelos")

tab_best, tab_alt, tab_comp = st.tabs(["Modelo Oficial (Best)", "Modelo Alternativo (Alt)", "Comparación"])

with tab_best:
    dashboard_model("Best", API_URL_BEST, MONITOR_FILE_BEST)

with tab_alt:
    dashboard_model("Alt", API_URL_ALT, MONITOR_FILE_ALT)

with tab_comp:
    st.header("📊 Comparación Best vs Alt")
    if MONITOR_FILE_BEST.exists() and MONITOR_FILE_ALT.exists():
        df_best = pd.read_csv(MONITOR_FILE_BEST)
        df_alt = pd.read_csv(MONITOR_FILE_ALT)

        df_best["modelo"] = "Best"
        df_alt["modelo"] = "Alt"
        df_all = pd.concat([df_best, df_alt])

        # Histograma comparativo
        fig = px.histogram(df_all, x="y_pred", color="modelo", barmode="overlay",
                           title="Distribución de Predicciones (Best vs Alt)")
        st.plotly_chart(fig, width="stretch")

        # Accuracy comparativo
        col1, col2 = st.columns(2)
        col1.metric("Accuracy Best", f"{accuracy_score(df_best['y_true'], df_best['y_pred']):.2f}")
        col2.metric("Accuracy Alt", f"{accuracy_score(df_alt['y_true'], df_alt['y_pred']):.2f}")
    else:
        st.warning(" Faltan logs para comparar ambos modelos.")
