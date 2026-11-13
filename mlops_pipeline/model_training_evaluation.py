"""
model_training_evaluation.py
Componente 2 del flujo MLOps: entrenamiento, evaluación y selección del mejor modelo.

Variable objetivo: Pago_atiempo (1 = pagó a tiempo, 0 = no pagó a tiempo).
Clase de interés para el negocio: 0 (los clientes que no pagan).

Criterios de selección:
- Performance: ROC-AUC promedio en validación cruzada.
- Consistency: baja variación entre folds y poca brecha entre train y validación.
- Scalability: tiempos de entrenamiento y predicción.
"""

# %% 1. Importaciones
import json
import time
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from ft_engineering import BASE_DIR, TARGET, construir_preprocesador, obtener_datos_modelado

from ft_engineering import BASE_DIR, construir_preprocesador, obtener_datos_modelado


# %% 2. Configuración
RANDOM_STATE = 42
UMBRAL_DECISION = 0.5
CLASE_INTERES = 0
ETIQUETAS = ["No pagó a tiempo (0)", "Pagó a tiempo (1)"]
MAX_BRECHA_SOBREAJUSTE = 0.10
RUTA_MODELO = BASE_DIR / "best_model.joblib"
RUTA_METADATA = BASE_DIR / "best_model_metadata.json"


# %% 3. summarize_classification: todas las métricas en un solo lugar
def summarize_classification(y_true, proba_pago, umbral: float = UMBRAL_DECISION) -> dict:
    """Calcula las métricas a partir de la probabilidad de pagar a tiempo (clase 1)."""
    y_pred = (proba_pago >= umbral).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_no_pago": precision_score(y_true, y_pred, pos_label=CLASE_INTERES, zero_division=0),
        "recall_no_pago": recall_score(y_true, y_pred, pos_label=CLASE_INTERES, zero_division=0),
        "f1_no_pago": f1_score(y_true, y_pred, pos_label=CLASE_INTERES, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba_pago),
        "pr_auc_no_pago": average_precision_score(y_true, 1 - proba_pago, pos_label=CLASE_INTERES),
    }


# %% 4. build_model: entrena, valida y mide un modelo
def build_model(nombre, estimador, X_train, y_train, X_test, y_test):
    """Arma el pipeline completo, hace validación cruzada, entrena y evalúa."""
    pipeline = Pipeline(steps=[
        ("preprocesador", construir_preprocesador()),
        ("modelo", estimador),
    ])

    # Validación cruzada estratificada (5 folds) para medir performance y consistencia
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    resultados_cv = cross_validate(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_auc = resultados_cv["test_score"]

    # Entrenamiento final sobre todo el set de entrenamiento y medición de tiempos
    inicio = time.perf_counter()
    pipeline.fit(X_train, y_train)
    tiempo_entrenamiento = time.perf_counter() - inicio

    # Predicción sobre el set de evaluación y medición de tiempos
    inicio = time.perf_counter()
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    tiempo_prediccion = time.perf_counter() - inicio

    # Métricas en train y test
    proba_train = pipeline.predict_proba(X_train)[:, 1]
    metricas_train = summarize_classification(y_train, proba_train)
    metricas_test = summarize_classification(y_test, proba_test)

    # Fila resumen con todas las métricas y tiempos
    fila = {
        "modelo": nombre,
        "cv_auc_media": cv_auc.mean(),
        "cv_auc_std": cv_auc.std(),
        "auc_train": metricas_train["roc_auc"],
        "brecha_sobreajuste": metricas_train["roc_auc"] - cv_auc.mean(),
        **{f"test_{metrica}": valor for metrica, valor in metricas_test.items()},
        "tiempo_entrenamiento_s": tiempo_entrenamiento,
        "tiempo_prediccion_ms_1000": tiempo_prediccion / len(X_test) * 1000 * 1000,
    }
    return pipeline, fila


# %% 5. Datos desde el componente de ingeniería de características
X_train, X_test, y_train, y_test, _ = obtener_datos_modelado()
print(f"Train: {X_train.shape} | no pagó a tiempo {(y_train == 0).mean():.2%}")
print(f"Test:  {X_test.shape} | no pagó a tiempo {(y_test == 0).mean():.2%}")


# %% 6. Modelos candidatos
modelos = {

    # Lista de modelos candidatos con hiperparámetros predefinidos.
    "Baseline (Dummy)": DummyClassifier(strategy="prior"),
    "Regresión Logística": LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
    ),
    "Árbol de Decisión": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=50, class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=50,
        class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=200, max_depth=3, min_samples_leaf=50,
        l2_regularization=1.0, class_weight="balanced", random_state=RANDOM_STATE,
    ),
}


# %% 7. Entrenamiento y evaluación de todos los modelos

# Se guardan los pipelines entrenados y las métricas en una lista de filas.
modelos_entrenados = {}
filas = []
for nombre, estimador in modelos.items():
    pipeline, fila = build_model(nombre, estimador, X_train, y_train, X_test, y_test)
    modelos_entrenados[nombre] = pipeline
    filas.append(fila)
    print(f"✔ {nombre}: AUC CV {fila['cv_auc_media']:.3f} ± {fila['cv_auc_std']:.3f}")


# %% 8. Tabla resumen de evaluación
tabla_resumen = (
    pd.DataFrame(filas)
    .set_index("modelo")
    .sort_values("cv_auc_media", ascending=False)
)
columnas_clave = [
    "cv_auc_media", "cv_auc_std", "auc_train", "brecha_sobreajuste",
    "test_roc_auc", "test_pr_auc_no_pago", "test_recall_no_pago",
    "test_precision_no_pago", "test_f1_no_pago", "test_accuracy",
    "tiempo_entrenamiento_s", "tiempo_prediccion_ms_1000",
]
print(tabla_resumen[columnas_clave].round(3).to_string())   


# %% 9. Gráficos comparativos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

tabla_grafico = tabla_resumen.sort_values("cv_auc_media")
axes[0].barh(tabla_grafico.index, tabla_grafico["cv_auc_media"],
             xerr=tabla_grafico["cv_auc_std"], color="steelblue")
axes[0].axvline(0.5, color="gray", linestyle="--", label="Azar (0.5)")
axes[0].set_title("Performance: ROC-AUC en validación cruzada")
axes[0].set_xlim(0.4, 1.0)
axes[0].legend()

posiciones = np.arange(len(tabla_grafico))
axes[1].barh(posiciones - 0.2, tabla_grafico["auc_train"], height=0.4, label="Train")
axes[1].barh(posiciones + 0.2, tabla_grafico["test_roc_auc"], height=0.4, label="Test")
axes[1].set_yticks(posiciones, tabla_grafico.index)
axes[1].set_title("Consistency: AUC train vs test")
axes[1].set_xlim(0.4, 1.0)
axes[1].legend()

axes[2].barh(tabla_grafico.index, tabla_grafico["tiempo_entrenamiento_s"], color="darkorange")
axes[2].set_title("Scalability: segundos de entrenamiento")

plt.tight_layout()
plt.show()



# %% 10. Curvas ROC y Precision-Recall de los modelos
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for nombre, pipeline in modelos_entrenados.items():
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, name=nombre, ax=axes[0])
    PrecisionRecallDisplay.from_estimator(
        pipeline, X_test, y_test, pos_label=CLASE_INTERES, name=nombre, ax=axes[1]
    )
axes[0].set_title("Curva ROC (test)")
axes[1].set_title("Curva Precision-Recall de la clase 0 (test)")
plt.tight_layout()
plt.show()


# %% 11. Selección del mejor modelo
def seleccionar_mejor_modelo(tabla: pd.DataFrame) -> str:
    """Mayor AUC en CV entre los modelos consistentes; en empate gana el más rápido."""
    # Se descarta el Dummy y se filtran los modelos con brecha de sobreajuste aceptable.
    candidatos = tabla.drop(index="Baseline (Dummy)")
    consistentes = candidatos[candidatos["brecha_sobreajuste"] <= MAX_BRECHA_SOBREAJUSTE]
    if consistentes.empty:
        consistentes = candidatos

    # Se selecciona el modelo con mayor AUC en CV; en caso de empate, el más rápido.
    mejor_auc = consistentes["cv_auc_media"].max()
    empatados = consistentes[consistentes["cv_auc_media"] >= mejor_auc - 0.01]
    return empatados["tiempo_entrenamiento_s"].idxmin()


nombre_mejor = seleccionar_mejor_modelo(tabla_resumen)
mejor_modelo = modelos_entrenados[nombre_mejor]
print(f"🏆 Mejor modelo: {nombre_mejor}")
print(tabla_resumen.loc[nombre_mejor, columnas_clave].round(3))


# %% 12. Evaluación detallada del mejor modelo

# Predicciones y métricas en test
proba_test = mejor_modelo.predict_proba(X_test)[:, 1]
y_pred_test = (proba_test >= UMBRAL_DECISION).astype(int)

print(classification_report(y_test, y_pred_test, target_names=ETIQUETAS))


# Matriz de confusion entre la clase real y la predicha
ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_test, display_labels=ETIQUETAS, cmap="Blues"
)
plt.title(f"Matriz de confusión: {nombre_mejor}")
plt.show()


# %% 13. Importancia de las variables
nombres_variables = mejor_modelo.named_steps["preprocesador"].named_steps["columnas"].get_feature_names_out()
estimador_final = mejor_modelo.named_steps["modelo"]

if hasattr(estimador_final, "coef_"):
    importancia = pd.Series(estimador_final.coef_[0], index=nombres_variables)
elif hasattr(estimador_final, "feature_importances_"):
    importancia = pd.Series(estimador_final.feature_importances_, index=nombres_variables)
else:
    importancia = None

if importancia is not None:
    top = importancia.reindex(importancia.abs().sort_values().index).tail(12)
    top.plot(kind="barh", figsize=(8, 6), color=np.where(top > 0, "seagreen", "indianred"))
    plt.title(f"Variables más influyentes: {nombre_mejor}")
    plt.tight_layout()
    plt.show()


# %% 14. Guardar el mejor modelo y su ficha técnica
joblib.dump(mejor_modelo, RUTA_MODELO)

metadata = {
    "modelo": nombre_mejor,
    "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
    "variable_objetivo": f"{TARGET} (1 = pagó a tiempo, 0 = no pagó a tiempo)",
    "clase_de_interes": CLASE_INTERES,
    "umbral_decision": UMBRAL_DECISION,
    "regla_decision": "se predice 1 (pagará) si la probabilidad de pago >= umbral",
    "filas_entrenamiento": int(len(X_train)),
    "metricas": {k: round(float(v), 4) for k, v in tabla_resumen.loc[nombre_mejor].items()},
}
with open(RUTA_METADATA, "w", encoding="utf-8") as archivo:
    json.dump(metadata, archivo, indent=2, ensure_ascii=False)

print(f"Modelo guardado en: {RUTA_MODELO}")