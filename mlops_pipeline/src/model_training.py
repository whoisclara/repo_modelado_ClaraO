import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import VotingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier

# ========================
# Directorios
# ========================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ========================
# Funciones utilitarias
# ========================
def evaluar_modelo(y_true, y_pred):
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    for clase in [0, 1]:
        summary[f"precision_{clase}"] = precision_score(y_true, y_pred, pos_label=clase, zero_division=0)
        summary[f"recall_{clase}"] = recall_score(y_true, y_pred, pos_label=clase, zero_division=0)
        summary[f"f1_{clase}"] = f1_score(y_true, y_pred, pos_label=clase, zero_division=0)
    return summary

def threshold_tuning(model, X_test, y_test, thresholds=None):
    if thresholds is None:
        thresholds = [i / 100 for i in range(5, 95, 5)]
    if not hasattr(model, "predict_proba"):
        return None
    probs = model.predict_proba(X_test)[:, 1]
    resultados = []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        resultados.append({
            "threshold": t,
            **evaluar_modelo(y_test, y_pred)
        })
    return pd.DataFrame(resultados)

def build_model(model, param_grid, X, y, resampling="none"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Se prioriza el recall de la clase 0
    scorer = make_scorer(recall_score, pos_label=0)

    steps = []
    if resampling == "smote":
        steps.append(("smote", SMOTE(random_state=42)))
    elif resampling == "smoteenn":
        steps.append(("smoteenn", SMOTEENN(random_state=42)))
    elif resampling == "smotetomek":
        steps.append(("smotetomek", SMOTETomek(random_state=42)))
    steps.append(("clf", model))
    pipe = ImbPipeline(steps)

    if param_grid:
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring=scorer, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
    else:
        pipe.fit(X_train, y_train)
        best_model = pipe
        best_params = None

    y_pred = best_model.predict(X_test)
    resultados = {
        "resampling": resampling,
        "cv_score": cross_val_score(best_model, X_train, y_train, cv=cv, scoring=scorer).mean(),
        "test_metrics": evaluar_modelo(y_test, y_pred),
        "threshold_tuning": threshold_tuning(best_model, X_test, y_test),
        "best_model": best_model,
        "best_params": best_params
    }
    return resultados

def main():
    X_scale = pd.read_csv(DATA_DIR / "X_train.csv")
    X_raw   = pd.read_csv(DATA_DIR / "X_train_raw.csv")
    y       = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale_pos_weight = n_neg / n_pos

    model_grids = { 
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1], "clf__solver": ["lbfgs"]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {"clf__n_estimators": [200], "clf__max_depth": [None, 20], "clf__max_features": ["sqrt"]}
        ),
        "BalancedRF": (
            BalancedRandomForestClassifier(random_state=42, n_jobs=-1),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 20]}
        ),
        "SVC": (
            SVC(probability=True, class_weight="balanced", random_state=42),
            {"clf__C": [0.1, 1], "clf__kernel": ["linear", "rbf"]}
        ),
        "LightGBM": (
            LGBMClassifier(random_state=42, class_weight="balanced", scale_pos_weight=scale_pos_weight),
            {"clf__num_leaves": [31], "clf__learning_rate": [0.05], "clf__n_estimators": [100]}
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42, scale_pos_weight=scale_pos_weight),
            {"clf__max_depth": [3], "clf__learning_rate": [0.05], "clf__n_estimators": [100]}
        )
}
    resampling_strategies = ["smote", "smoteenn", "smotetomek"]

    modelos_scale = ["LogisticRegression", "SVC"]
    modelos_noscale = ["RandomForest", "BalancedRF", "LightGBM", "XGBoost"]

    #Entrenar modelos
    resultados = {}
    for nombre, (modelo, grid) in model_grids.items():
        resultados[nombre] = {}
        for resample in resampling_strategies:
            print(f"\n>>> Entrenando {nombre} con {resample}")

            if nombre in modelos_scale:
                resultados[nombre][resample] = build_model(modelo, grid, X_scale, y, resampling=resample)
            else:
                resultados[nombre][resample] = build_model(modelo, grid, X_raw, y, resampling=resample)

    # Ensamble (voting)
    print("\n>>> Entrenando VotingClassifier")
    voting = VotingClassifier(estimators=[
        ("xgb", XGBClassifier(eval_metric="logloss", random_state=42)),
        ("lgbm", LGBMClassifier(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42))
    ], voting="soft")
    resultados["Voting"] = {"none": build_model(voting, None, X_raw, y)}

    resumen = []
    for nombre, configs in resultados.items():
        for resample, valores in configs.items():
            fila = {
                "Modelo": nombre,
                "Resampling": resample,
                "CV_Score_final": valores["cv_score"],
                "Recall_mora": valores["test_metrics"]["recall_0"],
                "Precision_mora": valores["test_metrics"]["precision_0"],
                "F1_mora": valores["test_metrics"]["f1_0"],
                "Accuracy": valores["test_metrics"]["accuracy"],
                "F1_weighted": valores["test_metrics"]["f1_weighted"],
                "Best_params": valores["best_params"]
            }
            resumen.append(fila)
    df_resumen = pd.DataFrame(resumen).sort_values(by="Recall_mora", ascending=False)

    df_resumen.to_csv(RESULTS_DIR / "resumen_modelos.csv", index=False)

    print("\n====================================================")
    print("RESUMEN DE TODOS LOS MODELOS (ordenado por Recall_mora)")
    print("====================================================")
    print(df_resumen.to_string(index=False))

    mejor_modelo = df_resumen.iloc[0]["Modelo"]
    mejor_resample = df_resumen.iloc[0]["Resampling"]
    print("====================================================")
    print(f" Mejor modelo seleccionado (máx. Recall en mora): {mejor_modelo} ({mejor_resample})")
    print("====================================================")

    joblib.dump(resultados[mejor_modelo][mejor_resample]["best_model"], MODEL_DIR / "mejor_modelo.pkl")
    print(f"\u2705 Guardado en {MODEL_DIR}/mejor_modelo.pkl")

    # ====================================================
    # Guardar modelo alternativo (Voting)
    # ====================================================
    if "Voting" in resultados:
        alt_model = resultados["Voting"]["none"]["best_model"]
        joblib.dump(alt_model, MODEL_DIR / "modelo_alternativo.pkl")
        print(f"✅ Guardado modelo alternativo en {MODEL_DIR}/modelo_alternativo.pkl")


    tt = resultados[mejor_modelo][mejor_resample]["threshold_tuning"]
    if tt is not None:
        tt.to_csv(RESULTS_DIR / "threshold_tuning_mejor.csv", index=False)
        print("\n=== Threshold tuning del mejor modelo guardado en results/threshold_tuning_mejor.csv ===")

if __name__ == "__main__":
    main()