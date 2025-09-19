import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse


from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # crea la carpeta si no existe

#========================================================================
#                        LIMPIEZA DE DATOS
#========================================================================

def limpia_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza inicial: reglas de validación, eliminar columnas redundantes, manejar valores faltantes, 
    ajustar rangos."""

    #Códigos de tipo credito con muy pocos registros
    df["tipo_credito"] = df["tipo_credito"].where(~df["tipo_credito"].isin([7, 68]), np.nan)

    # Convertir a NaN las edades mayores a 100
    df.loc[df["edad_cliente"] > 100, "edad_cliente"] = np.nan

    #Validar rango del puntaje de datacredito
    df["puntaje_datacredito"] = df["puntaje_datacredito"].where(
    (df["puntaje_datacredito"] >= 150) & (df["puntaje_datacredito"] <= 950),
    np.nan)

    #Imputación por filas
    fil=["puntaje_datacredito","saldo_mora", "edad_cliente","saldo_total", "tipo_credito"]
    df=df.dropna(subset=fil)

    # Variables con un alto valor de nulos
    df = df.drop("tendencia_ingresos", axis=1)

    ##Imputación de nulos para promedio_ingresos_dataredito con factores calculados en el EDA
    df["promedio_ingresos_datacredito"] = df["promedio_ingresos_datacredito"].replace(0, np.nan)

    factores = {"independiente": 0.25,"empleado": 0.72}

    def estimar_ingreso(row):
        if pd.isna(row["promedio_ingresos_datacredito"]):
            factor = factores.get(str(row["tipo_laboral"]).lower(), 1.0)
            return row["salario_cliente"] * factor
        else:
            return row["promedio_ingresos_datacredito"]
    
    df["promedio_ingresos_datacredito"] = df.apply(estimar_ingreso, axis=1)

    df = df.loc[~((df["promedio_ingresos_datacredito"] == 0) & (df["salario_cliente"] == 0))]
    
    #Imputar de variables redundantes
    df = df.drop(columns=["saldo_mora_codeudor", "saldo_principal", 
                 "puntaje", "creditos_sectorReal","salario_cliente"], errors="ignore")
    
    #Cambiar el tipo de credito a category
    df["tipo_credito"] = df["tipo_credito"].astype("category")

    return df

#========================================================================
#                          CREAR PIPELINES
#========================================================================

def define_type(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    var_bin = [c for c in cat_cols if X[c].nunique() == 2]
    var_poli = [c for c in cat_cols if X[c].nunique() > 2]
    var_num = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return var_bin, var_poli, var_num

def pipeline_noscale(var_bin, var_poli,var_num):
    return ColumnTransformer([
        ("bin", OneHotEncoder(drop="first", handle_unknown="ignore"), var_bin),
        ("poly", OneHotEncoder(handle_unknown="ignore"), var_poli),
        ("num", SimpleImputer(strategy="median"), var_num)])

def pipeline_scale(var_bin, var_poli,var_num):
    return ColumnTransformer([
        ("bin", OneHotEncoder(drop="first", handle_unknown="ignore"), var_bin),
        ("poly", OneHotEncoder(handle_unknown="ignore"), var_poli),
        ("num", MinMaxScaler(), var_num)])

#========================================================================
#                     TRANSFORMACIONES DE DATOS
#========================================================================

def transform_data(X,y,pipeline):
    """Aplicar transformaciones a los datos."""

    #Aplica el pipeline
    X_transformed = pipeline.fit_transform(X)

    # Convertir a array si es una matriz dispersa
    if issparse(X_transformed):
        X_transformed = X_transformed.toarray()

    # Obtener nombres de las columnas transformadas
    col_names= pipeline.get_feature_names_out()

    # Crear un DataFrame con las características transformadas
    X_df = pd.DataFrame(X_transformed, columns=col_names)

    df_transformed= pd.concat([X_df, y.reset_index(drop=True)], axis=1)

    return df_transformed

#========================================================================
#                      SPLIT TRAIN/TEST
#========================================================================

def split_data(df: pd.DataFrame, target: str, test_size: float=0.2, random_state: int=42):
    """Dividir los datos en conjuntos de entrenamiento y prueba."""
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test

#========================================================================
#                                  MAIN 
#========================================================================

def main(df: pd.DataFrame, target: str):

    df=limpia_data(df)

    X= df.drop(columns=[target])
    y= df[target]

    var_bin, var_poli, var_num= define_type(X)

    pipe_noscale= pipeline_noscale(var_bin, var_poli,var_num)
    pipe_scale= pipeline_scale(var_bin, var_poli,var_num)

    df_noscale= transform_data(X,y,pipe_noscale)
    df_scale= transform_data(X,y,pipe_scale)

    X_train, X_test, y_train, y_test= split_data(df_scale,target)

    df_noscale.to_csv(DATA_DIR / "df_sin_escalar.csv", index=False)
    df_scale.to_csv(DATA_DIR / "df_escalado.csv", index=False)



if __name__ == "__main__":
    
    df = pd.read_csv(DATA_DIR / "df_post_eda.csv")
    main(df, target="Pago_atiempo")
    

