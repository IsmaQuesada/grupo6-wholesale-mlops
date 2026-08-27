import joblib
import pandas as pd

from src.features.build_features import FeatureBuilder

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


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
