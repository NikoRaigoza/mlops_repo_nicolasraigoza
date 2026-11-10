"""Pruebas del componente de ingeniería de características."""

import numpy as np
import pandas as pd

import ft_engineering as fe


def test_cargar_datos_tiene_la_forma_esperada():
    df = fe.cargar_datos()
    assert df.shape == (10763, 23)
    assert pd.api.types.is_datetime64_any_dtype(df[fe.COLUMNA_FECHA])


def test_limpiar_datos_convierte_invalidos_en_nulos():
    df = pd.DataFrame({
        "edad_cliente": [122, 35],
        "salario_cliente": [0, 2_500_000],
        "puntaje_datacredito": [999, 780],
        "tendencia_ingresos": ["8315", "Creciente"],
        "tipo_credito": [4, 9],
        "puntaje": [95.2, 10.0],
    })
    limpio = fe.limpiar_datos(df)

    assert limpio.loc[0, ["edad_cliente", "salario_cliente", "puntaje_datacredito",
                          "tendencia_ingresos"]].isna().all()
    assert limpio.loc[1, "edad_cliente"] == 35
    assert "puntaje" not in limpio.columns
    assert limpio["tipo_credito"].tolist() == ["4", "9"]


def test_crear_variables_calcula_las_derivadas():
    df = pd.DataFrame({
        "cuota_pactada": [100.0],
        "salario_cliente": [1000.0],
        "total_otros_prestamos": [500.0],
        "promedio_ingresos_datacredito": [np.nan],
    })
    nuevo = fe.crear_variables(df)

    assert nuevo.loc[0, "carga_cuota_salario"] == 0.1
    assert nuevo.loc[0, "endeudamiento"] == 0.5
    assert nuevo.loc[0, "sin_info_buro"] == 1


def test_log_seguro_no_falla_con_negativos():
    resultado = fe.log_seguro(np.array([-5.0, 0.0, np.e - 1]))
    np.testing.assert_allclose(resultado, [0.0, 0.0, 1.0])


def test_dividir_datos_es_temporal_y_conserva_el_target():
    X_train, X_test, y_train, y_test = fe.dividir_datos(fe.cargar_datos())

    assert len(X_train) == 7713 and len(X_test) == 3050
    assert X_train[fe.COLUMNA_FECHA].max() < X_test[fe.COLUMNA_FECHA].min()
    assert fe.TARGET not in X_train.columns
    assert set(y_train.unique()) == {0, 1}


def test_preprocesador_entrega_matriz_limpia():
    X_train, _, _, _, preprocesador = fe.obtener_datos_modelado()
    matriz = preprocesador.fit_transform(X_train)

    assert matriz.shape == (7713, 25)
    assert not np.isnan(matriz).any()