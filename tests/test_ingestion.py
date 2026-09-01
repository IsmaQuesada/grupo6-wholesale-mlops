import pandas as pd
import pytest

from src.ingestion.ingest import EXPECTED_COLUMNS, EXPECTED_MIN_ROWS, compute_checksum, validate_schema


def test_validate_schema_ok(raw_data):
    """Valida que el dataset real pasa la validación estructural sin error."""
    validate_schema(raw_data)


def test_validate_schema_faltan_columnas():
    """Verifica que faltan columnas lanza ValueError."""
    df_incompleto = pd.DataFrame({"Fresh": [1], "Milk": [2]})
    with pytest.raises(ValueError, match="Faltan columnas"):
        validate_schema(df_incompleto)


def test_validate_schema_pocas_filas():
    """Verifica que pocas filas lanza ValueError."""
    df_pocas = pd.DataFrame({col: [0] for col in EXPECTED_COLUMNS})
    with pytest.raises(ValueError, match="menos de las"):
        validate_schema(df_pocas)


def test_compute_checksum_known_content(tmp_path):
    """Verifica que compute_checksum produce el SHA-256 correcto para contenido conocido."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")
    checksum = compute_checksum(test_file)
    # SHA-256 de "hello world" es conocido
    assert checksum == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
