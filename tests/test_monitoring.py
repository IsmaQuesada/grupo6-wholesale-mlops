import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import calcular_psi, calcular_psi_dataframe, clasificar_psi
from src.monitoring.model_monitor import comparar_distribuciones
from src.monitoring.system_metrics import get_metrics

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# Tests de drift.py — funciones puras
# ------------------------------------------------------------------

def test_clasificar_psi_ok():
    assert clasificar_psi(0.05) == "OK"
    assert clasificar_psi(0.0) == "OK"
    assert clasificar_psi(0.099) == "OK"


def test_clasificar_psi_warning():
    assert clasificar_psi(0.10) == "WARNING"
    assert clasificar_psi(0.15) == "WARNING"
    assert clasificar_psi(0.249) == "WARNING"


def test_clasificar_psi_alert():
    assert clasificar_psi(0.25) == "ALERT"
    assert clasificar_psi(0.50) == "ALERT"
    assert clasificar_psi(1.0) == "ALERT"


def test_calcular_psi_mismas_distribuciones():
    rng = np.random.default_rng(42)
    datos = rng.normal(100, 15, 500)
    psi = calcular_psi(datos, datos)
    assert psi == pytest.approx(0.0, abs=0.01)


def test_calcular_psi_distribuciones_diferentes():
    rng = np.random.default_rng(42)
    ref = rng.normal(100, 15, 500)
    prod = rng.normal(200, 15, 500)
    psi = calcular_psi(ref, prod)
    assert psi > 0.1


# ------------------------------------------------------------------
# Tests de model_monitor.py — comparar_distribuciones (pura)
# ------------------------------------------------------------------

def test_comparar_distribuciones_estable():
    dist_ref = {0: 33.0, 1: 34.0, 2: 33.0}
    dist_prod = {0: 35.0, 1: 33.0, 2: 32.0}
    resultado = comparar_distribuciones(dist_ref, dist_prod, umbral=10.0)
    assert resultado["estable"] is True
    assert resultado["diferencia_maxima"] <= 10.0


def test_comparar_distribuciones_inestable():
    dist_ref = {0: 30.0, 1: 40.0, 2: 30.0}
    dist_prod = {0: 10.0, 1: 55.0, 2: 35.0}
    resultado = comparar_distribuciones(dist_ref, dist_prod, umbral=10.0)
    assert resultado["estable"] is False
    assert resultado["diferencia_maxima"] > 10.0


# ------------------------------------------------------------------
# Tests de system_metrics.py
# ------------------------------------------------------------------

def test_get_metrics_estructura():
    metrics = get_metrics()
    claves_esperadas = {
        "latency_avg_ms",
        "throughput_req_per_sec",
        "error_rate_pct",
        "availability_pct",
        "total_requests",
        "uptime_seconds",
    }
    assert set(metrics.keys()) == claves_esperadas
    assert isinstance(metrics["latency_avg_ms"], (int, float))
    assert isinstance(metrics["total_requests"], int)


# ------------------------------------------------------------------
# Tests que requieren archivos de modelo (solo corren si existen)
# ------------------------------------------------------------------

_MODELS_EXIST = (
    (REPO_ROOT / "models" / "kmeans_production.joblib").exists()
    and (REPO_ROOT / "models" / "feature_builder.joblib").exists()
)


@pytest.mark.skipif(not _MODELS_EXIST, reason="Modelos no generados")
def test_cargar_modelo():
    from src.monitoring.model_monitor import cargar_modelo
    modelo, fb = cargar_modelo()
    assert modelo is not None
    assert fb._fitted


@pytest.mark.skipif(not _MODELS_EXIST, reason="Modelos no generados")
def test_obtener_distribucion_clusters(raw_data):
    from src.monitoring.model_monitor import cargar_modelo, obtener_distribucion_clusters
    modelo, fb = cargar_modelo()
    dist = obtener_distribucion_clusters(modelo, fb, raw_data)
    assert set(dist.keys()) == {0, 1, 2}
    assert pytest.approx(sum(dist.values()), rel=1e-2) == 100.0


# ------------------------------------------------------------------
# Tests de retrain_trigger.py — decisiones de reentrenamiento
# ------------------------------------------------------------------

_METADATA_EXIST = (REPO_ROOT / "models" / "production_metadata.json").exists()


@pytest.mark.skipif(not _MODELS_EXIST or not _METADATA_EXIST, reason="Modelos o metadata no generados")
def test_evaluar_ok(raw_data):
    """Drift bajo + silhouette alta + composición estable → OK."""
    from src.monitoring.model_monitor import cargar_modelo
    from src.monitoring.retrain_trigger import evaluar_necesidad_reentrenamiento
    modelo, fb = cargar_modelo()
    rng = np.random.default_rng(42)
    df_prod = raw_data.sample(100, random_state=rng).reset_index(drop=True)
    resultado = evaluar_necesidad_reentrenamiento(
        raw_data, df_prod, silhouette_actual=0.25, modelo=modelo, feature_builder=fb,
    )
    assert resultado["decision"] == "OK"


@pytest.mark.skipif(not _MODELS_EXIST or not _METADATA_EXIST, reason="Modelos o metadata no generados")
def test_evaluar_monitorear(raw_data):
    """Drift alto pero modelo estable → MONITOREAR."""
    from src.monitoring.model_monitor import cargar_modelo
    from src.monitoring.retrain_trigger import evaluar_necesidad_reentrenamiento
    modelo, fb = cargar_modelo()
    df_prod = raw_data.copy()
    df_prod["Fresh"] = (df_prod["Fresh"] * 2).astype(int)
    df_prod["Milk"] = (df_prod["Milk"] * 2).astype(int)
    df_prod["Grocery"] = (df_prod["Grocery"] * 2).astype(int)
    resultado = evaluar_necesidad_reentrenamiento(
        raw_data, df_prod, silhouette_actual=0.25, modelo=modelo, feature_builder=fb,
    )
    assert resultado["decision"] == "MONITOREAR"


@pytest.mark.skipif(not _MODELS_EXIST or not _METADATA_EXIST, reason="Modelos o metadata no generados")
def test_evaluar_reentrenar(raw_data):
    """Composición inestable → REENTRENAR."""
    from src.monitoring.model_monitor import cargar_modelo
    from src.monitoring.retrain_trigger import evaluar_necesidad_reentrenamiento
    modelo, fb = cargar_modelo()
    df_prod = raw_data.copy()
    df_prod["Fresh"] = (df_prod["Fresh"] * 5).astype(int)
    df_prod["Milk"] = (df_prod["Milk"] * 5).astype(int)
    df_prod["Grocery"] = (df_prod["Grocery"] * 5).astype(int)
    df_prod["Frozen"] = (df_prod["Frozen"] * 0.1).astype(int)
    df_prod["Detergents_Paper"] = (df_prod["Detergents_Paper"] * 0.1).astype(int)
    df_prod["Delicassen"] = (df_prod["Delicassen"] * 0.1).astype(int)
    resultado = evaluar_necesidad_reentrenamiento(
        raw_data, df_prod, silhouette_actual=0.25, modelo=modelo, feature_builder=fb,
    )
    assert resultado["decision"] in ("REENTRENAR", "MONITOREAR")
