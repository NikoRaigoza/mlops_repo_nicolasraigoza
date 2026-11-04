"""
ft_engineering.py
Componente 1 del flujo MLOps: ingeniería de características.

Responsabilidades:
1. Cargar los datos crudos.
2. Aplicar las reglas de limpieza que se definieron en el EDA.
3. Crear las variables derivadas.
4. Construir el preprocesador (ColumnTransformer con 3 ramas).
5. Separar los conjuntos de entrenamiento y evaluación.
"""

# %% 1. Importaciones
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)





# %% 2. Configuración del proyecto
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

RUTA_DATOS = BASE_DIR.parent / "Base_de_datos.csv"

TARGET_ORIGINAL = "Pago_atiempo"
TARGET = "impago"
COLUMNA_FECHA = "fecha_prestamo"
FECHA_CORTE = "2025-06-01"

COLUMNAS_A_ELIMINAR = ["puntaje", "saldo_mora_codeudor", "saldo_principal"]
CATEGORIAS_TENDENCIA = ["Decreciente", "Estable", "Creciente"]
REGLAS_RANGO = {
    "edad_cliente": (18, 100),
    "salario_cliente": (1, 100_000_000),
    "puntaje_datacredito": (150, 950),
}

COLUMNAS_NUMERICAS = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    "carga_cuota_salario",
    "endeudamiento",
    "sin_info_buro",
]
COLUMNAS_CATEGORICAS = ["tipo_credito", "tipo_laboral"]
COLUMNAS_ORDINALES = ["tendencia_ingresos"]





# %% 3. Carga de datos
def cargar_datos(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Lee el CSV crudo y convierte la fecha del préstamo a datetime."""
    df = pd.read_csv(ruta, parse_dates=[COLUMNA_FECHA])
    return df




# %% 4. Limpieza: reglas de validación del EDA
def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica los valores inválidos como NaN y elimina las variables descartadas."""
    df = df.copy()
    df = df.drop(columns=[c for c in COLUMNAS_A_ELIMINAR if c in df.columns])

    for columna, (minimo, maximo) in REGLAS_RANGO.items():
        fuera_de_rango = ~df[columna].between(minimo, maximo)
        df.loc[fuera_de_rango, columna] = np.nan

    tendencia_invalida = ~df["tendencia_ingresos"].isin(CATEGORIAS_TENDENCIA)
    df.loc[tendencia_invalida, "tendencia_ingresos"] = np.nan

    df["tipo_credito"] = df["tipo_credito"].astype(str)
    return df





# %% 5. Variables derivadas
def crear_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las variables calculadas propuestas en el EDA."""
    df = df.copy()
    df["carga_cuota_salario"] = df["cuota_pactada"] / df["salario_cliente"]
    df["endeudamiento"] = df["total_otros_prestamos"] / df["salario_cliente"]
    df["sin_info_buro"] = df["promedio_ingresos_datacredito"].isna().astype(int)
    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza + variables derivadas en un solo paso (se usa dentro del pipeline)."""
    return crear_variables(limpiar_datos(df))





# %% 6. Preprocesador: ColumnTransformer con 3 ramas
def log_seguro(X):
    """Logaritmo log(1 + x) que nunca falla: los negativos se llevan a 0 antes."""
    return np.log1p(np.clip(X, 0, None))


def construir_preprocesador() -> Pipeline:
    """Devuelve el pipeline de preprocesamiento (sin modelo)."""
    pipeline_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(log_seguro, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])

    pipeline_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=30,
            sparse_output=False,
        )),
    ])

    pipeline_ordinal = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[CATEGORIAS_TENDENCIA],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", pipeline_numerico, COLUMNAS_NUMERICAS),
            ("categoric", pipeline_categorico, COLUMNAS_CATEGORICAS),
            ("categoric_ordinales", pipeline_ordinal, COLUMNAS_ORDINALES),
        ],
        remainder="drop",
    )

    preprocesador = Pipeline(steps=[
        ("preparacion", FunctionTransformer(preparar_datos)),
        ("columnas", column_transformer),
    ])
    return preprocesador




# %% 7. Separación temporal en entrenamiento y evaluación
def dividir_datos(df: pd.DataFrame, fecha_corte: str = FECHA_CORTE):
    """Entrena con el pasado y evalúa con el futuro (validación out-of-time)."""
    df = df.sort_values(COLUMNA_FECHA).reset_index(drop=True)

    y = (df[TARGET_ORIGINAL] == 0).astype(int).rename(TARGET)
    X = df.drop(columns=[TARGET_ORIGINAL])

    es_entrenamiento = df[COLUMNA_FECHA] < pd.Timestamp(fecha_corte)

    X_train, y_train = X[es_entrenamiento], y[es_entrenamiento]
    X_test, y_test = X[~es_entrenamiento], y[~es_entrenamiento]
    return X_train, X_test, y_train, y_test





# %% 8. Función principal del componente
def obtener_datos_modelado():
    """Ejecuta el componente completo y retorna los conjuntos y el preprocesador."""
    df = cargar_datos()
    X_train, X_test, y_train, y_test = dividir_datos(df)
    preprocesador = construir_preprocesador()
    return X_train, X_test, y_train, y_test, preprocesador





# %% 9. Ejecución y verificación
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocesador = obtener_datos_modelado()

    print(f"Entrenamiento: {X_train.shape[0]} filas | tasa de impago {y_train.mean():.2%}")
    print(f"Evaluación:    {X_test.shape[0]} filas | tasa de impago {y_test.mean():.2%}")

    X_train_transformado = preprocesador.fit_transform(X_train)
    nombres = preprocesador.named_steps["columnas"].get_feature_names_out()
    print(f"Matriz final de entrenamiento: {X_train_transformado.shape}")
    print(pd.DataFrame(X_train_transformado, columns=nombres).head())