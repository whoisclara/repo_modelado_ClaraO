import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Directorios
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)


#========================================================================
#                        Funciones Utilitarias
#========================================================================

def evaluar_modelo(y_true, y_pred):
    """Genera un resumen de las métricas de clasificación. Clase 0 = risk(no paga a tiempo)
      y Clase 1 = no risk (paga a tiempo)"""
    report = classification_report(y_true, y_pred, output_dict=True)

    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
   
    for clase in [0, 1]:
        summary[f"precision_{clase}"] = precision_score(y_true, y_pred, pos_label=clase)
        summary[f"recall_{clase}"]    = recall_score(y_true, y_pred, pos_label=clase)
        summary[f"f1_{clase}"]        = f1_score(y_true, y_pred, pos_label=clase)

    return summary

def precision_recall_mix(y_true, y_pred):
    """Métrica mixta: prioriza la precisión en morosos, pero mantiene recall > 0"""
    p = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    r = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return 0.7*p + 0.3*r   # 70% peso precisión, 30% peso recall

def build_model(model, X, y):
    """Construye y evalúa un modelo con y sin SMOTE (split 70/30 + cross-validation)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    resultados={}
    pr_mix=make_scorer(precision_recall_mix, greater_is_better=True)
    #precision_mora = make_scorer(precision_score, pos_label=0, zero_division=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Sin SMOTE
    pipeline_nosmote = ImbPipeline(steps=[('classifier', model)])
    scores_no = cross_val_score(pipeline_nosmote, X_train, y_train, cv=cv, scoring=pr_mix) #Entrenamiento
    pipeline_nosmote.fit(X_train, y_train)
    y_pred_no = pipeline_nosmote.predict(X_test)

    #SMOTE
    pipe_smote = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf", model)
    ])

    ##Entrenamiento del modelo con SMOTE
    scores_smote = cross_val_score(pipe_smote, X_train, y_train, cv=cv, scoring=pr_mix)
    pipe_smote.fit(X_train, y_train)
    ##Predicciones
    y_pred_smote = pipe_smote.predict(X_test)


    resultados["sin_smote"] = {
        "cv_mean_score_final": scores_no.mean(),
        "test_metrics": evaluar_modelo(y_test, y_pred_no)
    }

    resultados["con_smote"] = {
        "cv_mean_score_final": scores_smote.mean(),
        "test_metrics": evaluar_modelo(y_test, y_pred_smote)
    }

    return resultados

def main():
    df_scale= pd.read_csv(DATA_DIR / "df_escalado.csv")
    df_noscale= pd.read_csv(DATA_DIR / "df_sin_escalar.csv")

    target="Pago_atiempo"
    y= df_scale[target]
    X_scale= df_scale.drop(columns=[target])
    X_raw= df_noscale.drop(columns=[target])

    #Definir modelos
    model_scale={
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    }

    model_noscale={
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "BalancedRF": BalancedRandomForestClassifier(n_estimators=200, random_state=42),   
        "EasyEnsemble": EasyEnsembleClassifier(n_estimators=50, random_state=42) 
    }

    #Entrenar y evaluar modelos
    resultados = {}
    for nombre, modelo in model_scale.items():
        print(f"\n>>> Entrenando {nombre} (dataset escalado)")
        resultados[nombre] = build_model(modelo, X_scale, y)

    for nombre, modelo in model_noscale.items():
        print(f"\n>>> Entrenando {nombre} (dataset sin escalar)")
        resultados[nombre] = build_model(modelo, X_raw, y)

    # 4. Construir tabla resumen
    resumen = []
    for nombre, configs in resultados.items():
        for setting, valores in configs.items():  # sin_smote / con_smote
            fila = {
                "Modelo": nombre,
                "Pipeline": setting,
                "CV_Precision_mora": valores["cv_mean_precision_mora"],
                "Recall_mora": valores["test_metrics"]["recall_0"],
                "Precision_mora": valores["test_metrics"]["precision_0"],
                "F1_mora": valores["test_metrics"]["f1_0"],
                "Accuracy": valores["test_metrics"]["accuracy"],
                "F1_weighted": valores["test_metrics"]["f1_weighted"],
            }
            resumen.append(fila)
    df_resumen = pd.DataFrame(resumen).sort_values(by="Precision_mora", ascending=False)

    #Seleccionar y guardar el mejor modelo
    mejor_modelo = df_resumen.iloc[0]["Modelo"]
    mejor_pipeline = df_resumen.iloc[0]["Pipeline"]

    print("\n====================================================")
    print("RESUMEN DE MODELOS (TOP 5)")
    print("====================================================")
    print(df_resumen.to_string(index=False))

    print("\n====================================================")
    print(f"Mejor modelo seleccionado: {mejor_modelo} ({mejor_pipeline})")
    print("====================================================")

    # Guardar pipeline entrenado final
    joblib.dump(resultados[mejor_modelo][mejor_pipeline], MODEL_DIR / f"{mejor_modelo}_{mejor_pipeline}.pkl")

    return

if __name__ == "__main__":
    main()