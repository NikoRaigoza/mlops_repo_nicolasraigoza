"""Prueba de integración: ejecuta el entrenamiento completo y revisa sus resultados."""

import json
import runpy

import numpy as np
import pytest

from conftest import CARPETA_PIPELINE


@pytest.fixture(scope="module")
def entrenamiento():
    return runpy.run_path(str(CARPETA_PIPELINE / "model_training_evaluation.py"), run_name="__main__")


def test_summarize_classification_mide_la_clase_cero(entrenamiento):
    resumen = entrenamiento["summarize_classification"]
    y_real = np.array([0, 0, 1, 1])
    proba_pago = np.array([0.2, 0.7, 0.8, 0.9])

    metricas = resumen(y_real, proba_pago)

    assert metricas["recall_no_pago"] == 0.5
    assert metricas["precision_no_pago"] == 1.0
    assert metricas["roc_auc"] == 1.0


def test_se_comparan_varios_modelos(entrenamiento):
    tabla = entrenamiento["tabla_resumen"]
    assert len(tabla) == 5
    assert {"test_roc_auc", "test_recall_no_pago", "test_accuracy"} <= set(tabla.columns)


def test_se_guarda_el_mejor_modelo_y_su_ficha(entrenamiento):
    assert entrenamiento["RUTA_MODELO"].is_file()
    with open(entrenamiento["RUTA_METADATA"], encoding="utf-8") as archivo:
        metadata = json.load(archivo)

    assert metadata["modelo"] == entrenamiento["nombre_mejor"]
    assert metadata["clase_de_interes"] == 0