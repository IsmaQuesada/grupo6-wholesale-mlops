from pathlib import Path

import pytest
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def raw_data():
    """Carga el dataset crudo para tests de datos."""
    return pd.read_csv(REPO_ROOT / "data" / "raw" / "wholesale_customers_raw.csv")


@pytest.fixture
def sample_request():
    """Request de ejemplo para tests del API."""
    return {
        "Fresh": 12669,
        "Milk": 9656,
        "Grocery": 7561,
        "Frozen": 214,
        "Detergents_Paper": 2674,
        "Delicassen": 1338,
    }
