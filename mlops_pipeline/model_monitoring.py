"""
model_monitoring.py
Componente 3 del flujo MLOps: monitoreo del modelo y detección de data drift.

Dos formas de ejecutarlo:
- Como trabajo periódico (job):  python mlops_pipeline/model_monitoring.py
- Como tablero en Streamlit:      streamlit run mlops_pipeline/model_monitoring.py
"""

# %% 1. Importaciones
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import roc_auc_score
from streamlit import runtime

from ft_engineering import (
    BASE_DIR,
    COLUMNA_FECHA,
    COLUMNAS_CATEGORICAS,
    COLUMNAS_NUMERICAS,
    COLUMNAS_ORDINALES,
    TARGET,
    cargar_datos,
    dividir_datos,
    preparar_datos,
)


# %% 2. Configuración del monitoreo
RUTA_MODELO = BASE_DIR / "best_model.joblib"
RUTA_METADATA = BASE_DIR / "best_model_metadata.json"
RUTA_PREDICCIONES = BASE_DIR / "predicciones_log.csv"
RUTA_REPORTE = BASE_DIR / "reporte_drift.csv"

PSI_MODERADO = 0.10
PSI_CRITICO = 0.25
P_VALOR_ALERTA = 0.05
TAMANO_MUESTRA = 500
MIN_FILAS_PERIODO = 100
MAX_VALORES_DISCRETOS = 10
SEMILLA = 42

VARIABLES_NUMERICAS = COLUMNAS_NUMERICAS + ["probabilidad_pago"]
VARIABLES_CATEGORICAS = COLUMNAS_CATEGORICAS + COLUMNAS_ORDINALES


# %% 3. Tabla de datos + pronósticos del modelo
def generar_tabla_predicciones():
    """Une cada registro con el pronóstico del modelo (referencia y producción)."""
    modelo = joblib.load(RUTA_MODELO)
    with open(RUTA_METADATA, encoding="utf-8") as archivo:
        umbral = json.load(archivo)["umbral_decision"]

    df = cargar_datos()
    X_train, X_test, y_train, y_test = dividir_datos(df)

    tablas = {}
    for nombre, X, y in [("referencia", X_train, y_train), ("produccion", X_test, y_test)]:
        tabla = preparar_datos(X)
        tabla["probabilidad_pago"] = modelo.predict_proba(X)[:, 1]
        tabla["prediccion"] = (tabla["probabilidad_pago"] >= umbral).astype(int)
        tabla[TARGET] = y.to_numpy()
        tabla["periodo"] = tabla[COLUMNA_FECHA].dt.to_period("M").astype(str)
        tablas[nombre] = tabla

    tablas["produccion"].to_csv(RUTA_PREDICCIONES, index=False)
    return tablas["referencia"], tablas["produccion"]


# %% 4. Convertir dos muestras en conteos comparables
def obtener_conteos(referencia: pd.Series, actual: pd.Series, es_categorica: bool):
    """Cuenta referencia y actual sobre las MISMAS cajas (rangos o categorías)."""
    referencia = referencia.dropna()
    actual = actual.dropna()

    if not es_categorica and referencia.nunique() > MAX_VALORES_DISCRETOS:
        cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, 11)))
        if len(cortes) >= 3:
            conteo_ref, _ = np.histogram(np.clip(referencia, cortes[0], cortes[-1]), bins=cortes)
            conteo_act, _ = np.histogram(np.clip(actual, cortes[0], cortes[-1]), bins=cortes)
            return conteo_ref.astype(float), conteo_act.astype(float)
        referencia = referencia > cortes[0]
        actual = actual > cortes[0]

    categorias = sorted(set(referencia.astype(str)) | set(actual.astype(str)))
    conteo_ref = referencia.astype(str).value_counts().reindex(categorias, fill_value=0)
    conteo_act = actual.astype(str).value_counts().reindex(categorias, fill_value=0)
    return conteo_ref.to_numpy(dtype=float), conteo_act.to_numpy(dtype=float)


# %% 5. Métricas de data drift
def calcular_psi(conteo_ref, conteo_act) -> float:
    """Population Stability Index."""
    p_ref = np.clip(conteo_ref / conteo_ref.sum(), 1e-6, None)
    p_act = np.clip(conteo_act / conteo_act.sum(), 1e-6, None)
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def calcular_jensen_shannon(conteo_ref, conteo_act) -> float:
    """Distancia de Jensen-Shannon en base 2 (0 = iguales, 1 = totalmente distintas)."""
    return float(jensenshannon(conteo_ref, conteo_act, base=2))


def prueba_chi_cuadrado(conteo_ref, conteo_act) -> float:
    """p-valor de Chi-cuadrado sobre la tabla de contingencia referencia vs actual."""
    tabla = np.array([conteo_ref, conteo_act])
    tabla = tabla[:, tabla.sum(axis=0) > 0]
    if tabla.shape[1] < 2:
        return 1.0
    return float(chi2_contingency(tabla).pvalue)


def clasificar_psi(psi: float) -> str:
    """Semáforo según los umbrales estándar de la industria."""
    if psi >= PSI_CRITICO:
        return "🔴 Crítico"
    if psi >= PSI_MODERADO:
        return "🟡 Moderado"
    return "🟢 Estable"


# %% 6. Drift de todas las variables entre dos conjuntos
def calcular_drift(referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Calcula PSI, Jensen-Shannon y la prueba estadística de cada variable."""
    filas = []
    for variable in VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS:
        es_categorica = variable in VARIABLES_CATEGORICAS
        conteo_ref, conteo_act = obtener_conteos(referencia[variable], actual[variable], es_categorica)

        if es_categorica:
            prueba = "Chi-cuadrado"
            p_valor = prueba_chi_cuadrado(conteo_ref, conteo_act)
        else:
            prueba = "Kolmogorov-Smirnov"
            p_valor = float(ks_2samp(referencia[variable].dropna(), actual[variable].dropna()).pvalue)

        psi = calcular_psi(conteo_ref, conteo_act)
        filas.append({
            "variable": variable,
            "tipo": "Categórica" if es_categorica else "Numérica",
            "psi": psi,
            "jensen_shannon": calcular_jensen_shannon(conteo_ref, conteo_act),
            "prueba": prueba,
            "p_valor": p_valor,
            "diferencia_significativa": p_valor < P_VALOR_ALERTA,
            "estado": clasificar_psi(psi),
        })
    return pd.DataFrame(filas)


# %% 7. Muestreo periódico: un análisis por cada mes de producción
def monitorear_por_periodo(referencia: pd.DataFrame, produccion: pd.DataFrame) -> pd.DataFrame:
    """Para cada periodo toma una muestra y la compara contra la referencia."""
    resultados = []
    for periodo, datos in produccion.groupby("periodo"):
        if len(datos) < MIN_FILAS_PERIODO:
            continue
        muestra = datos.sample(n=min(len(datos), TAMANO_MUESTRA), random_state=SEMILLA)
        drift = calcular_drift(referencia, muestra)
        drift["periodo"] = periodo
        drift["n_muestra"] = len(muestra)
        resultados.append(drift)
    return pd.concat(resultados, ignore_index=True)


def desempeno_por_periodo(produccion: pd.DataFrame) -> pd.DataFrame:
    """Compara lo que el modelo predijo con lo que realmente pasó, mes a mes."""
    filas = []
    for periodo, datos in produccion.groupby("periodo"):
        tiene_ambas_clases = datos[TARGET].nunique() == 2
        filas.append({
            "periodo": periodo,
            "registros": len(datos),
            "no_pago_real_%": (datos[TARGET] == 0).mean() * 100,
            "no_pago_predicho_%": (datos["prediccion"] == 0).mean() * 100,
            "roc_auc": roc_auc_score(datos[TARGET], datos["probabilidad_pago"])
            if tiene_ambas_clases else np.nan,
        })
    return pd.DataFrame(filas)


# %% 8. Análisis temporal: tendencias y cambios abruptos
def tabla_psi_temporal(historial: pd.DataFrame) -> pd.DataFrame:
    """Matriz periodo x variable con el PSI de cada mes."""
    return historial.pivot(index="periodo", columns="variable", values="psi").sort_index()


def detectar_cambios_abruptos(historial: pd.DataFrame, salto: float = PSI_MODERADO) -> pd.DataFrame:
    """Periodos donde el PSI de una variable sube más que `salto` frente al mes anterior."""
    aumentos = tabla_psi_temporal(historial).diff().stack().rename("aumento_psi").reset_index()
    return aumentos[aumentos["aumento_psi"] >= salto].sort_values("aumento_psi", ascending=False)


def calcular_tendencias(historial: pd.DataFrame) -> pd.DataFrame:
    """Pendiente del PSI en el tiempo: positiva = el drift viene creciendo."""
    psi_temporal = tabla_psi_temporal(historial)
    x = np.arange(len(psi_temporal))
    pendientes = {
        variable: np.polyfit(x, psi_temporal[variable].to_numpy(), 1)[0]
        for variable in psi_temporal.columns
    }
    tendencias = pd.Series(pendientes, name="pendiente_psi").sort_values(ascending=False).reset_index()
    tendencias = tendencias.rename(columns={"index": "variable"})
    tendencias["tendencia"] = np.where(tendencias["pendiente_psi"] > 0.005, "🔺 En aumento", "➖ Estable")
    return tendencias



# %% 9. Alertas y recomendaciones automáticas
def generar_alertas(drift_periodo: pd.DataFrame):
    """Devuelve mensajes por variable y una recomendación general."""
    criticas = drift_periodo[drift_periodo["psi"] >= PSI_CRITICO]
    moderadas = drift_periodo[drift_periodo["psi"].between(PSI_MODERADO, PSI_CRITICO, inclusive="left")]

    mensajes = []
    for _, fila in criticas.iterrows():
        mensajes.append(("error", f"🔴 {fila['variable']}: PSI {fila['psi']:.3f}. "
                                  "La población cambió de forma significativa."))
    for _, fila in moderadas.iterrows():
        mensajes.append(("warning", f"🟡 {fila['variable']}: PSI {fila['psi']:.3f}. "
                                    "Cambio moderado, conviene vigilarla."))

    psi_prediccion = drift_periodo.loc[drift_periodo["variable"] == "probabilidad_pago", "psi"].iloc[0]
    if psi_prediccion >= PSI_CRITICO or len(criticas) >= 3:
        recomendacion = ("error", "🔁 Se recomienda REENTRENAR el modelo con datos recientes: "
                                  "la población o las predicciones cambiaron demasiado.")
    elif len(criticas) > 0 or len(moderadas) > 0:
        recomendacion = ("warning", "🔎 Se recomienda REVISAR las variables señaladas y su "
                                    "origen (cambios en el proceso, errores de captura, nuevos clientes).")
    else:
        recomendacion = ("success", "✅ Sin drift relevante. El modelo puede seguir operando.")
    return mensajes, recomendacion



# %% 10. Gráficos
def grafico_barras_riesgo(drift_periodo: pd.DataFrame):
    """Barras horizontales de PSI con los umbrales del semáforo."""
    datos = drift_periodo.sort_values("psi")
    colores = np.select(
        [datos["psi"] >= PSI_CRITICO, datos["psi"] >= PSI_MODERADO],
        ["tab:red", "gold"],
        default="tab:green",
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(datos["variable"], datos["psi"], color=colores)
    ax.axvline(PSI_MODERADO, color="gold", linestyle="--", label=f"Moderado ({PSI_MODERADO})")
    ax.axvline(PSI_CRITICO, color="tab:red", linestyle="--", label=f"Crítico ({PSI_CRITICO})")
    ax.set_xlabel("PSI")
    ax.legend()
    fig.tight_layout()
    return fig


def grafico_distribucion(referencia: pd.DataFrame, actual: pd.DataFrame, variable: str):
    """Distribución histórica (referencia) vs actual de una variable."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if variable in VARIABLES_CATEGORICAS:
        proporciones = pd.DataFrame({
            "Histórico": referencia[variable].value_counts(normalize=True),
            "Actual": actual[variable].value_counts(normalize=True),
        }).fillna(0)
        proporciones.plot(kind="bar", ax=ax)
        ax.set_ylabel("Proporción")
    else:
        limite = referencia[variable].quantile(0.99)
        rango = (referencia[variable].min(), limite)
        ax.hist(referencia[variable].dropna(), bins=30, range=rango, density=True, alpha=0.5, label="Histórico")
        ax.hist(actual[variable].dropna(), bins=30, range=rango, density=True, alpha=0.5, label="Actual")
        ax.set_ylabel("Densidad")
        ax.legend()
    ax.set_title(f"Histórico vs actual: {variable}")
    fig.tight_layout()
    return fig


def grafico_evolucion(psi_temporal: pd.DataFrame, variables: list):
    """Evolución del PSI en el tiempo con las líneas de umbral."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for variable in variables:
        ax.plot(psi_temporal.index, psi_temporal[variable], marker="o", label=variable)
    ax.axhline(PSI_MODERADO, color="gold", linestyle="--")
    ax.axhline(PSI_CRITICO, color="tab:red", linestyle="--")
    ax.set_ylabel("PSI")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig



# %% 11. Aplicación Streamlit
def mostrar(fig):
    """Muestra una figura en Streamlit y la cierra para no acumular memoria."""
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data
def cargar_resultados():
    """Calcula todo una sola vez; Streamlit guarda el resultado en caché."""
    referencia, produccion = generar_tabla_predicciones()
    historial = monitorear_por_periodo(referencia, produccion)
    desempeno = desempeno_por_periodo(produccion)
    return referencia, produccion, historial, desempeno


def ejecutar_app():
    st.set_page_config(page_title="Monitoreo de Data Drift", page_icon="📊", layout="wide")
    st.title("📊 Monitoreo del modelo de pago a tiempo")
    st.caption("Histórico = datos con los que se entrenó el modelo · Actual = créditos posteriores al corte")

    referencia, produccion, historial, desempeno = cargar_resultados()

    periodos = sorted(historial["periodo"].unique())
    periodo = st.sidebar.selectbox("Periodo a analizar", periodos, index=len(periodos) - 1)
    st.sidebar.markdown(
        f"**Umbrales PSI**\n\n🟢 Estable: < {PSI_MODERADO}\n\n"
        f"🟡 Moderado: {PSI_MODERADO} – {PSI_CRITICO}\n\n🔴 Crítico: ≥ {PSI_CRITICO}"
    )
    st.sidebar.caption(f"Muestra por periodo: hasta {TAMANO_MUESTRA} registros. "
                       f"Se omiten periodos con menos de {MIN_FILAS_PERIODO}.")

    drift_periodo = historial[historial["periodo"] == periodo]
    datos_periodo = produccion[produccion["periodo"] == periodo]
    fila_desempeno = desempeno.set_index("periodo").loc[periodo]
    psi_prediccion = drift_periodo.loc[drift_periodo["variable"] == "probabilidad_pago", "psi"].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variables críticas 🔴", int((drift_periodo["psi"] >= PSI_CRITICO).sum()))
    col2.metric("Variables moderadas 🟡",
                int(drift_periodo["psi"].between(PSI_MODERADO, PSI_CRITICO, inclusive="left").sum()))
    col3.metric("PSI de la predicción", f"{psi_prediccion:.3f}")
    col4.metric("ROC-AUC del periodo",
                "N/A" if pd.isna(fila_desempeno["roc_auc"]) else f"{fila_desempeno['roc_auc']:.3f}")

    tab_drift, tab_dist, tab_tiempo, tab_alertas = st.tabs(
        ["🚦 Drift por variable", "📈 Distribuciones", "⏱️ Evolución temporal", "🔔 Alertas y recomendaciones"]
    )

    with tab_drift:
        st.subheader(f"Métricas de drift por variable · {periodo}")
        columnas = ["estado", "variable", "tipo", "psi", "jensen_shannon", "prueba", "p_valor"]
        st.dataframe(
            drift_periodo[columnas].sort_values("psi", ascending=False).round(4),
            hide_index=True, width="stretch",
        )
        mostrar(grafico_barras_riesgo(drift_periodo))

    with tab_dist:
        variable = st.selectbox("Variable", VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS)
        mostrar(grafico_distribucion(referencia, datos_periodo, variable))

    with tab_tiempo:
        psi_temporal = tabla_psi_temporal(historial)
        por_defecto = psi_temporal.max().sort_values(ascending=False).head(5).index.tolist()
        seleccion = st.multiselect("Variables", psi_temporal.columns.tolist(), default=por_defecto)
        if seleccion:
            mostrar(grafico_evolucion(psi_temporal, seleccion))

        col_a, col_b = st.columns(2)
        col_a.markdown("**Cambios abruptos** (el PSI sube más de "
                       f"{PSI_MODERADO} frente al mes anterior)")
        col_a.dataframe(detectar_cambios_abruptos(historial).round(3), hide_index=True)
        col_b.markdown("**Tendencias del drift**")
        col_b.dataframe(calcular_tendencias(historial).round(4), hide_index=True)

        st.markdown("**Desempeño real del modelo por periodo**")
        st.dataframe(desempeno.round(3), hide_index=True, width="stretch")

    with tab_alertas:
        mensajes, (tipo, texto) = generar_alertas(drift_periodo)
        getattr(st, tipo)(texto)
        for tipo_mensaje, mensaje in mensajes:
            getattr(st, tipo_mensaje)(mensaje)



# %% 12. Trabajo periódico (sin interfaz)
def ejecutar_job():
    referencia, produccion = generar_tabla_predicciones()
    historial = monitorear_por_periodo(referencia, produccion)
    historial.to_csv(RUTA_REPORTE, index=False)

    ultimo_periodo = historial["periodo"].max()
    mensajes, (_, recomendacion) = generar_alertas(historial[historial["periodo"] == ultimo_periodo])

    print(f"Predicciones guardadas en: {RUTA_PREDICCIONES}")
    print(f"Reporte de drift guardado en: {RUTA_REPORTE}")
    print(f"\nÚltimo periodo analizado: {ultimo_periodo}")
    for _, mensaje in mensajes:
        print(" ", mensaje)
    print(recomendacion)


# %% 13. Punto de entrada
if __name__ == "__main__":
    if runtime.exists():
        ejecutar_app()
    else:
        ejecutar_job()


