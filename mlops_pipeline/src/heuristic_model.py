import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class HeuristicModel(BaseEstimator, ClassifierMixin):
    def __init__(self, puntaje_threshold=650, cuota_ingreso_threshold= 0.4, mora_threshold=0, 
                 huella_threshold=6, creditos_threshold=8, salario_threshold=5_000_000):

        self.puntaje_threshold = puntaje_threshold
        self.cuota_ingreso_threshold = cuota_ingreso_threshold
        self.mora_threshold = mora_threshold
        self.huella_threshold = huella_threshold        
        self.creditos_threshold = creditos_threshold
        self.salario_threshold = salario_threshold

    def fit (self, X, y=None):
        if y is not None:
            self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        predictions = []

        for _, row in X.iterrows():
            acepta = True

            #Reglas heurísticas

            #Regla 1: Puntaje de datacredito bajo -> riesgo
            if row["num__puntaje_datacredito"] < self.puntaje_threshold:
                acepta = False

            #Regla 2: Relación cuota/ingreso alta -> riesgo
            if row["num__plazo_meses"] > 0 and row["num__promedio_ingresos_datacredito"] > 0:
                cuota = row["num__capital_prestado"] / row["num__plazo_meses"]
                rb = cuota / row["num__promedio_ingresos_datacredito"]
                if rb > self.cuota_ingreso_threshold:
                    acepta = False

            #Regla 3: Saldo en mora -> riesgo
            if row["num__saldo_mora"] > self.mora_threshold:
                acepta = False

            #Regla 4: Huella de consulta crediticia alta -> riesgo
            if row["num__huella_consulta"] > self.huella_threshold:
                acepta = False

            #Regla 5: Créditos vigentes altos -> riesgo
            if row["num__cant_creditosvigentes"] > self.creditos_threshold:
                acepta = False

            #Regla 6: Ingresos altos y buen puntaje -> no riesgo

            if (row["num__promedio_ingresos_datacredito"] >= self.salario_threshold and
                row["num__puntaje_datacredito"] >= self.puntaje_threshold):
                acepta = True   

            predictions.append(1 if acepta else 0)


        return np.array(predictions)

#========================================================================
#                     CARGAR DATASET
#========================================================================

df=pd.read_csv(DATA_DIR / "df_sin_escalar.csv")
X=df.drop(["Pago_atiempo"], axis=1)
y=df["Pago_atiempo"]

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#========================================================================
#                     ENTRENAR MODELO
#========================================================================

#Entrenar y evaluar en test
model = HeuristicModel()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

test_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    #"confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

#métricas en train # Calcular métricas en train
model_pipe = Pipeline(steps=[("heuristic", HeuristicModel())])
y_train_pred = model_pipe.fit(X_train, y_train).predict(X_train)

train_metrics = {
    "accuracy": accuracy_score(y_train, y_train_pred),
    "precision": precision_score(y_train, y_train_pred),
    "recall": recall_score(y_train, y_train_pred),
    "f1": f1_score(y_train, y_train_pred),
}

#Validación cruzada
scoring_metrics = ["accuracy", "precision", "recall", "f1"]
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

model_pipe = Pipeline(steps=[("heuristic", HeuristicModel())])
cv_results = {metric: cross_val_score(model_pipe, X, y, cv=kfold, scoring=metric)
              for metric in scoring_metrics}
cv_results_df = pd.DataFrame(cv_results)

means = cv_results_df.mean()
stds = cv_results_df.std()

#========================================================================
#                             GRÁFICAS
#========================================================================

#Train vs Cross-Validation
plt.figure(figsize=(8, 6))
x_pos = range(len(scoring_metrics))
plt.bar(x_pos, [train_metrics[m] for m in scoring_metrics], width=0.4, label="Train")
plt.bar([i + 0.4 for i in x_pos], [test_metrics[m] for m in scoring_metrics],
        width=0.4, label="Test")
plt.xticks([i + 0.2 for i in x_pos], scoring_metrics)
plt.ylabel("Score")
plt.title("Métricas Train vs Test")
plt.legend()
plt.show()

#Curva de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(
    model_pipe, X, y, cv=kfold, scoring="f1"
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train F1")
plt.plot(train_sizes, test_scores.mean(axis=1), "o-", label="CV F1")
plt.title("Curva de Aprendizaje - HeuristicModel")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

