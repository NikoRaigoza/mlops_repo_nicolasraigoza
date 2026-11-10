"""Pruebas del monitoreo de data drift y de la app de Streamlit."""

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import model_monitoring as mm
from conftest import CARPETA_PIPELINE


def test_psi_es_cero_para_distribuciones_iguales():
    conteo = np.array([10.0, 20.0, 30.0])
    assert mm.calcular_psi(conteo, conteo) < 1e-9


def test_psi_detecta_un_cambio_fuerte():
    psi = mm.calcular_psi(np.array([90.0, 10.0]), np.array([10.0, 90.0]))
    assert psi > mm.PSI_CRITICO
    assert mm.clasificar_psi(psi) == "🔴 Crítico"


def test_semaforo_tiene_tres_niveles():
    assert mm.clasificar_psi(0.05) == "🟢 Estable"
    assert mm.clasificar_psi(0.15) == "🟡 Moderado"


def test_conteos_de_variable_con_muchos_ceros_no_quedan_en_una_caja():
    referencia = pd.Series([0.0] * 95 + list(range(1, 50)))
    actual = pd.Series([0.0] * 50 + list(range(1, 51)))
    conteo_ref, conteo_act = mm.obtener_conteos(referencia, actual, es_categorica=False)
    assert len(conteo_ref) == len(conteo_act) >= 2


def test_chi_cuadrado_y_jensen_shannon():
    iguales = np.array([50.0, 50.0])
    assert mm.prueba_chi_cuadrado(iguales, iguales) > 0.05
    assert mm.calcular_jensen_shannon(iguales, iguales) < 1e-9
    assert mm.prueba_chi_cuadrado(np.array([90.0, 10.0]), np.array([10.0, 90.0])) < 0.05


def test_job_genera_reporte_y_recomendacion():
    referencia, produccion = mm.generar_tabla_predicciones()
    historial = mm.monitorear_por_periodo(referencia, produccion)
    _, (tipo, _) = mm.generar_alertas(historial[historial["periodo"] == historial["periodo"].max()])

    assert {"psi", "jensen_shannon", "p_valor", "estado"} <= set(historial.columns)
    assert historial["n_muestra"].min() >= mm.MIN_FILAS_PERIODO
    assert tipo in {"error", "warning", "success"}

    mm.ejecutar_job()
    assert mm.RUTA_REPORTE.is_file() and mm.RUTA_PREDICCIONES.is_file()


def test_app_streamlit_carga_sin_errores():
    app = AppTest.from_file(str(CARPETA_PIPELINE / "model_monitoring.py"), default_timeout=180)
    app.run()

    assert not app.exception
    assert len(app.metric) == 4
    assert len(app.tabs) == 4