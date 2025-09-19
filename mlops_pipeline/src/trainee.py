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


def threshold_tuning(model, X_val, y_val, recall_max=0.95):
    """
    Busca thresholds que cumplan recall_0 <= recall_max (95% por defecto).
    Entre ellos elige el mejor por F1_0.
    """
    if not hasattr(model, "predict_proba"):
        return None, None

    p1 = model.predict_proba(X_val)[:, 1]
    thresholds = [i / 100 for i in range(1, 100)]
    rows = []

    for t in thresholds:
        y_pred = (p1 >= t).astype(int)
        m = evaluar_modelo(y_val, y_pred)
        rows.append({"threshold": t, **m})

    df_thr = pd.DataFrame(rows)

    # Filtrar thresholds que no superen recall_max
    factibles = df_thr[df_thr["recall_0"] <= recall_max]

    if len(factibles) == 0:
        print("⚠️ Ningún threshold cumple recall <= 95%. Usando mejor recall global.")
        best_row = df_thr.sort_values("recall_0", ascending=False).iloc[0]
    else:
        # Elegir el threshold con mayor f1_0 entre los factibles
        best_row = factibles.sort_values("f1_0", ascending=False).iloc[0]

    best_t = float(best_row["threshold"])
    return df_thr, best_t


def build_model(model, param_grid, X, y, resampling="none"):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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

    # Tuning de threshold con restricción recall <= 95%
    tt_df, best_t = threshold_tuning(best_model, X_val, y_val, recall_max=0.95)

    # Aplicar threshold seleccionado
    if hasattr(best_model, "predict_proba") and best_t is not None:
        p1 = best_model.predict_proba(X_val)[:, 1]
        y_pred = (p1 >= best_t).astype(int)
    else:
        y_pred = best_model.predict(X_val)

    resultados = {
        "resampling": resampling,
        "cv_score": cross_val_score(best_model, X_train, y_train, cv=cv, scoring=scorer).mean(),
        "test_metrics": evaluar_modelo(y_val, y_pred),
        "threshold_tuning": tt_df,
        "best_threshold": best_t,
        "best_model": best_model,
        "best_params": best_params
    }
    return resultados


def main():
    df_scale = pd.read_csv(DATA_DIR / "df_escalado.csv")
    df_noscale = pd.read_csv(DATA_DIR / "df_sin_escalar.csv")

    target = "Pago_atiempo"
    y = df_scale[target]
    X_scale = df_scale.drop(columns=[target])
    X_raw = df_noscale.drop(columns=[target])

    n_pos = y.value_counts()[1]
    n_neg = y.value_counts()[0]
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

    resultados = {}
    for nombre, (modelo, grid) in model_grids.items():
        resultados[nombre] = {}
        for resample in resampling_strategies:
            print(f"\n>>> Entrenando {nombre} con {resample}")
            resultados[nombre][resample] = build_model(modelo, grid, X_raw, y, resampling=resample)

    # Ensamble (Voting)
    print("\n>>> Entrenando VotingClassifier")
    voting = VotingClassifier(estimators=[
        ("xgb", XGBClassifier(eval_metric="logloss", random_state=42)),
        ("lgbm", LGBMClassifier(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42))
    ], voting="soft")
    resultados["Voting"] = {"none": build_model(voting, None, X_raw, y)}

    # Resumen
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
                "Best_params": valores["best_params"],
                "Best_threshold": valores["best_threshold"]
            }
            resumen.append(fila)
    df_resumen = pd.DataFrame(resumen).sort_values(by="Recall_mora", ascending=False)
    df_resumen.to_csv(RESULTS_DIR / "resumen_modelos.csv", index=False)

    print("\n====================================================")
    print("RESUMEN DE TODOS LOS MODELOS (recall <= 95%)")
    print("====================================================")
    print(df_resumen.to_string(index=False))

    mejor_modelo = df_resumen.iloc[0]["Modelo"]
    mejor_resample = df_resumen.iloc[0]["Resampling"]

    print("\n====================================================")
    print(f"🏆 Mejor modelo seleccionado (recall ≤ 95% y mejor F1_0): {mejor_modelo} ({mejor_resample})")
    print("====================================================")

    best = resultados[mejor_modelo][mejor_resample]
    joblib.dump(best["best_model"], MODEL_DIR / "mejor_modelo.pkl")
    with open(RESULTS_DIR / "best_threshold.txt", "w") as f:
        f.write(str(best.get("best_threshold", 0.5)))

    if best["threshold_tuning"] is not None:
        best["threshold_tuning"].to_csv(RESULTS_DIR / "threshold_tuning_mejor.csv", index=False)


if __name__ == "__main__":
    main()
