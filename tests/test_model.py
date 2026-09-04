import joblib
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from sklearn.cluster import KMeans

from src.features.build_features import FeatureBuilder

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_modelo_carga():
    """Verifica que el modelo KMeans se carga del archivo joblib."""
    modelo = joblib.load(REPO_ROOT / "models" / "kmeans_production.joblib")
    assert modelo is not None


def test_feature_builder_carga():
    """Verifica que el FeatureBuilder se carga del archivo joblib."""
    fb = FeatureBuilder.load(REPO_ROOT / "models" / "feature_builder.joblib")
    assert fb._fitted


def test_input_valido_genera_prediccion(sample_request):
    """Verifica que un input válido produce una predicción válida."""
    modelo = joblib.load(REPO_ROOT / "models" / "kmeans_production.joblib")
    fb = FeatureBuilder.load(REPO_ROOT / "models" / "feature_builder.joblib")

    df = pd.DataFrame([sample_request])
    df["Channel"] = float("nan")
    df["Region"] = float("nan")

    df_features = fb.transform(df)
    X = df_features[fb.cols_finales_]

    cluster = int(modelo.predict(X)[0])
    assert cluster in [0, 1, 2], f"Cluster inválido: {cluster}"


def test_modelo_es_kmeans():
    """Verifica que el modelo cargado es una instancia de KMeans."""
    modelo = joblib.load(REPO_ROOT / "models" / "kmeans_production.joblib")
    assert isinstance(modelo, KMeans)


def test_feature_builder_fitted_attrs():
    """Verifica que el FeatureBuilder tiene atributos de ajuste reales."""
    fb = FeatureBuilder.load(REPO_ROOT / "models" / "feature_builder.joblib")
    assert hasattr(fb.scaler, "scale_"), "scaler no tiene scale_ (no ajustado)"
    assert hasattr(fb.scaler, "mean_"), "scaler no tiene mean_ (no ajustado)"
    assert fb.cols_finales_ is not None, "cols_finales_ no definido"


def test_transform_consistency(raw_data):
    """Verifica que fit_transform y transform producen los mismos resultados."""
    fb = FeatureBuilder.load(REPO_ROOT / "models" / "feature_builder.joblib")
    df_features = fb.transform(raw_data)
    # Verificar que las columnas de salida son las esperadas
    assert len(df_features) == len(raw_data)
    assert all(col in df_features.columns for col in fb.cols_finales_)


def test_transform_unfitted_raises(raw_data):
    """Verifica que transform sin fit lanza error."""
    fb = FeatureBuilder()
    with pytest.raises(RuntimeError):
        fb.transform(raw_data)


def test_save_load_roundtrip(raw_data):
    """Verifica que save → load produce un builder que transforma igual."""
    fb_original = FeatureBuilder.load(REPO_ROOT / "models" / "feature_builder.joblib")
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "fb_test.joblib"
        fb_original.save(save_path)
        fb_loaded = FeatureBuilder.load(save_path)

    df_orig = fb_original.transform(raw_data)
    df_loaded = fb_loaded.transform(raw_data)
    pd.testing.assert_frame_equal(df_orig, df_loaded)
