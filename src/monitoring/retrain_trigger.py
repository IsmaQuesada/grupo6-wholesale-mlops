"""
retrain_trigger.py — Lógica de decisión de reentrenamiento (Sección R).

Combina tres señales, no solo una: drift de datos (PSI máximo entre las 6
variables), degradación de calidad de cluster (silhouette vs. línea base) y
estabilidad de composición de clusters (Sección O3). Esta tercera señal es
necesaria porque el silhouette por sí solo puede no detectar cambios reales:
un modelo puede seguir produciendo clusters bien separados geométricamente
aunque la composición de qué clientes caen en cada cluster haya cambiado
significativamente (caso observado en BATCH 3 durante las pruebas).

Drift alto por sí solo NO implica reentrenar: la Sección P mostró que BATCH 1
y BATCH 2, con drift leve/moderado, mantuvieron el modelo estable. Solo se
reentrena cuando hay evidencia de degradación real (caída de silhouette o
inestabilidad de composición), no solo de cambio de distribución de entrada.
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.features.build_features import COLS_GASTO
from src.monitoring.drift import calcular_psi_dataframe
from src.monitoring.model_monitor import comparar_distribuciones, obtener_distribucion_clusters

UMBRAL_PSI = 0.25  # mismo umbral ALERT usado en la Sección P
RATIO_SILHOUETTE_MINIMO = 0.70  # tolera hasta 30% de caída respecto a la línea base
UMBRAL_ESTABILIDAD_PP = 10.0  # mismo umbral ya usado en model_monitor.py (Sección O3)


def _silhouette_base() -> float:
    """Lee el silhouette del modelo actualmente en producción."""
    with open(REPO_ROOT / "models" / "production_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata["silhouette"]


def evaluar_necesidad_reentrenamiento(
    df_referencia: pd.DataFrame,
    df_produccion: pd.DataFrame,
    silhouette_actual: float,
    modelo,
    feature_builder,
    umbral_psi: float = UMBRAL_PSI,
    ratio_silhouette_minimo: float = RATIO_SILHOUETTE_MINIMO,
    umbral_estabilidad_pp: float = UMBRAL_ESTABILIDAD_PP,
) -> dict:
    """
    REENTRENAR si hay degradación real (silhouette bajo O composición de clusters
    inestable). Si solo hay drift de entrada sin ninguna de esas dos señales ->
    MONITOREAR. Si nada de lo anterior -> OK.
    """
    psi_por_variable = calcular_psi_dataframe(df_referencia, df_produccion, COLS_GASTO)
    psi_max = max(psi_por_variable.values())

    silhouette_base = _silhouette_base()
    umbral_silhouette_absoluto = silhouette_base * ratio_silhouette_minimo

    dist_referencia = obtener_distribucion_clusters(modelo, feature_builder, df_referencia)
    dist_produccion = obtener_distribucion_clusters(modelo, feature_builder, df_produccion)
    comparacion = comparar_distribuciones(dist_referencia, dist_produccion, umbral_estabilidad_pp)

    drift_alto = psi_max > umbral_psi
    performance_baja = silhouette_actual < umbral_silhouette_absoluto
    modelo_inestable = not comparacion["estable"]

    if performance_baja or modelo_inestable:
        decision = "REENTRENAR"
    elif drift_alto:
        decision = "MONITOREAR"
    else:
        decision = "OK"

    return {
        "decision": decision,
        "psi_max": round(psi_max, 4),
        "silhouette_actual": round(silhouette_actual, 4),
        "umbral_silhouette": round(umbral_silhouette_absoluto, 4),
        "diferencia_composicion_pp": comparacion["diferencia_maxima"],
        "drift_alto": drift_alto,
        "performance_baja": performance_baja,
        "modelo_inestable": modelo_inestable,
    }


if __name__ == "__main__":
    from src.monitoring.model_monitor import cargar_modelo

    csv_path = REPO_ROOT / "data" / "raw" / "wholesale_customers_raw.csv"
    if not csv_path.exists():
        print("ERROR: No existe data/raw/wholesale_customers_raw.csv")
        print("  Ejecuta primero: python src/ingestion/ingest.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    modelo, fb = cargar_modelo()

    # Smoke test: evaluar con los mismos datos (deberia dar OK)
    resultado = evaluar_necesidad_reentrenamiento(
        df, df, silhouette_actual=0.25, modelo=modelo, feature_builder=fb,
    )
    print(f"Smoke test (misma referencia y produccion):")
    print(f"  Decision: {resultado['decision']}")
    print(f"  PSI max:  {resultado['psi_max']}")
    print(f"  Drift:    {resultado['drift_alto']}")
    print(f"  Perf:     {resultado['performance_baja']}")
    print(f"  Estable:  {not resultado['modelo_inestable']}")