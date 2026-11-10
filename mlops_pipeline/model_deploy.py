"""
model_deploy.py
Componente 4 del flujo MLOps: despliegue del mejor modelo como API (FastAPI).

Endpoints:
- GET  /health        -> estado del servicio y del modelo cargado.
- POST /predict       -> predicción por lotes; recibe JSON con una lista de registros.
- POST /predict/csv   -> predicción por lotes; recibe un archivo CSV.

Ejecución local (desde la raíz del repositorio):
    uvicorn model_deploy:app --app-dir mlops_pipeline --reload
"""

# %% 1. Importaciones
import io
import json
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ft_engineering import BASE_DIR, TARGET


# %% 2. Configuración
RUTA_MODELO = BASE_DIR / "best_model.joblib"
RUTA_METADATA = BASE_DIR / "best_model_metadata.json"
MAX_REGISTROS = 20_000
MAX_BYTES_CSV = 5 * 1024 * 1024

COLUMNAS_NUMERICAS_ENTRADA = [
    "tipo_credito", "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito", "cant_creditosvigentes",
    "huella_consulta", "saldo_mora", "saldo_total", "saldo_principal", "saldo_mora_codeudor",
    "creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]
COLUMNAS_TEXTO_ENTRADA = ["fecha_prestamo", "tipo_laboral", "tendencia_ingresos"]
COLUMNAS_ENTRADA = COLUMNAS_NUMERICAS_ENTRADA + COLUMNAS_TEXTO_ENTRADA


# %% 3. Carga del modelo y de su ficha técnica
def cargar_modelo():
    """Carga el pipeline entrenado (limpieza + preprocesador + modelo) y su metadata."""
    modelo = joblib.load(RUTA_MODELO)
    with open(RUTA_METADATA, encoding="utf-8") as archivo:
        metadata = json.load(archivo)
    return modelo, metadata


modelo, metadata = cargar_modelo()


# %% 4. Esquema de entrada: un crédito (datos crudos, como en Base_de_datos.csv)
class Credito(BaseModel):
    """Un registro de solicitud de crédito. Todos los campos aceptan nulos."""

    tipo_credito: Optional[int] = None
    fecha_prestamo: Optional[str] = None
    capital_prestado: Optional[float] = None
    plazo_meses: Optional[int] = None
    edad_cliente: Optional[int] = None
    tipo_laboral: Optional[str] = None
    salario_cliente: Optional[float] = None
    total_otros_prestamos: Optional[float] = None
    cuota_pactada: Optional[float] = None
    puntaje_datacredito: Optional[float] = None
    cant_creditosvigentes: Optional[int] = None
    huella_consulta: Optional[int] = None
    saldo_mora: Optional[float] = None
    saldo_total: Optional[float] = None
    saldo_principal: Optional[float] = None
    saldo_mora_codeudor: Optional[float] = None
    creditos_sectorFinanciero: Optional[int] = None
    creditos_sectorCooperativo: Optional[int] = None
    creditos_sectorReal: Optional[int] = None
    promedio_ingresos_datacredito: Optional[float] = None
    tendencia_ingresos: Optional[str] = None


class SolicitudPrediccion(BaseModel):
    """Cuerpo del POST /predict: una lista de créditos (predicción por lotes)."""

    registros: list[Credito] = Field(..., min_length=1, max_length=MAX_REGISTROS)



# %% 5. Lógica de predicción (compartida por JSON y CSV)
def preparar_entrada(df: pd.DataFrame) -> pd.DataFrame:
    """Deja el DataFrame con las columnas y tipos que espera el modelo."""
    df = df.reindex(columns=COLUMNAS_ENTRADA)
    for columna in COLUMNAS_NUMERICAS_ENTRADA:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    return df


def predecir(df: pd.DataFrame) -> dict:
    """Aplica el pipeline completo y devuelve las predicciones en formato estructurado."""
    entrada = preparar_entrada(df)
    umbral = metadata["umbral_decision"]
    probabilidades = modelo.predict_proba(entrada)[:, 1]

    predicciones = []
    for indice, probabilidad in enumerate(probabilidades):
        prediccion = int(probabilidad >= umbral)
        predicciones.append({
            "indice": indice,
            "probabilidad_pago": round(float(probabilidad), 4),
            "prediccion": prediccion,
            "resultado": "Pagará a tiempo" if prediccion == 1 else "No pagará a tiempo",
        })

    return {
        "modelo": metadata["modelo"],
        "variable_objetivo": metadata["variable_objetivo"],
        "umbral_decision": umbral,
        "n_registros": len(predicciones),
        "predicciones": predicciones,
    }


# %% 6. API con FastAPI
app = FastAPI(
    title="API de predicción de pago a tiempo",
    description="Predice Pago_atiempo (1 = pagó a tiempo, 0 = no pagó) por lotes.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    """Verifica que el servicio esté arriba y qué modelo tiene cargado."""
    return {
        "estado": "ok",
        "modelo": metadata["modelo"],
        "fecha_entrenamiento": metadata["fecha_entrenamiento"],
    }


@app.post("/predict")
def predict(solicitud: SolicitudPrediccion) -> dict:
    """Predicción por lotes a partir de JSON."""
    df = pd.DataFrame([registro.model_dump() for registro in solicitud.registros])
    return predecir(df)


@app.post("/predict/csv")
async def predict_csv(archivo: UploadFile = File(...)) -> dict:
    """Predicción por lotes a partir de un archivo CSV con el formato de Base_de_datos.csv."""
    contenido = await archivo.read()
    if len(contenido) > MAX_BYTES_CSV:
        raise HTTPException(status_code=413, detail="El archivo supera el tamaño máximo de 5 MB.")

    try:
        df = pd.read_csv(io.BytesIO(contenido))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=400, detail="El archivo no es un CSV válido.") from error

    if df.empty or len(df) > MAX_REGISTROS:
        raise HTTPException(status_code=400, detail=f"El CSV debe tener entre 1 y {MAX_REGISTROS} filas.")

    df = df.drop(columns=[TARGET], errors="ignore")
    return predecir(df)


