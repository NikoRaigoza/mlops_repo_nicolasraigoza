"""Pruebas de la API de predicción (FastAPI)."""

import io
import json

import pandas as pd
from fastapi.testclient import TestClient

from conftest import RAIZ
from model_deploy import app

cliente = TestClient(app)
MUESTRA = pd.read_csv(RAIZ / "Base_de_datos.csv").head(5)


def test_health_responde_ok():
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "ok"


def test_predict_json_por_lotes():
    registros = json.loads(MUESTRA.drop(columns=["Pago_atiempo"]).to_json(orient="records"))
    respuesta = cliente.post("/predict", json={"registros": registros})
    cuerpo = respuesta.json()

    assert respuesta.status_code == 200
    assert cuerpo["n_registros"] == 5
    assert all(p["prediccion"] in (0, 1) for p in cuerpo["predicciones"])
    assert all(0 <= p["probabilidad_pago"] <= 1 for p in cuerpo["predicciones"])


def test_predict_csv_da_lo_mismo_que_json():
    registros = json.loads(MUESTRA.drop(columns=["Pago_atiempo"]).to_json(orient="records"))
    por_json = cliente.post("/predict", json={"registros": registros}).json()

    archivo = io.BytesIO(MUESTRA.to_csv(index=False).encode("utf-8"))
    por_csv = cliente.post("/predict/csv", files={"archivo": ("lote.csv", archivo, "text/csv")}).json()

    assert por_csv["predicciones"] == por_json["predicciones"]


def test_predict_acepta_registros_incompletos():
    respuesta = cliente.post("/predict", json={"registros": [{"edad_cliente": 30}]})
    assert respuesta.status_code == 200


def test_predict_rechaza_entradas_invalidas():
    assert cliente.post("/predict", json={"registros": []}).status_code == 422
    assert cliente.post("/predict", json={"registros": [{"edad_cliente": "abc"}]}).status_code == 422

    vacio = io.BytesIO(b"")
    assert cliente.post("/predict/csv", files={"archivo": ("x.csv", vacio, "text/csv")}).status_code == 400