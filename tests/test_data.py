from src.data_quality.validate import COLS_GASTO, validar_calidad_datos

EXPECTED_COLUMNS = [
    "Channel", "Region", "Fresh", "Milk", "Grocery",
    "Frozen", "Detergents_Paper", "Delicassen",
]


def test_esquema_columnas(raw_data):
    """Verifica que existen las 8 columnas esperadas (sin importar orden)."""
    assert set(raw_data.columns) == set(EXPECTED_COLUMNS)


def test_tipos_numericos(raw_data):
    """Verifica que todas las columnas son numéricas."""
    for col in EXPECTED_COLUMNS:
        assert raw_data[col].dtype in ["int64", "float64"], f"{col} no es numérica"


def test_rango_channel(raw_data):
    """Verifica que Channel solo tiene valores 1 o 2."""
    assert set(raw_data["Channel"].unique()).issubset({1, 2})


def test_rango_region(raw_data):
    """Verifica que Region solo tiene valores 1, 2 o 3."""
    assert set(raw_data["Region"].unique()).issubset({1, 2, 3})


def test_sin_nulos(raw_data):
    """Verifica que no hay valores nulos."""
    assert raw_data.isnull().sum().sum() == 0


def test_sin_gastos_negativos(raw_data):
    """Verifica que ningún gasto es negativo."""
    assert (raw_data[COLS_GASTO] >= 0).all().all()


def test_minimo_400_filas(raw_data):
    """Verifica que hay al menos 400 registros."""
    assert len(raw_data) >= 400


def test_sin_duplicados(raw_data):
    """Verifica que la proporción de duplicados está bajo el umbral de 2%."""
    prop_duplicados = raw_data.duplicated().mean()
    assert prop_duplicados < 0.02, f"{prop_duplicados:.2%} duplicados (umbral: 2%)"


def test_esquema_8_columnas_exactas(raw_data):
    """Verifica que no hay columnas fuera del esquema esperado."""
    assert set(raw_data.columns) == set(EXPECTED_COLUMNS)


def test_filas_no_excesivas(raw_data):
    """Sanity check: el dataset no tiene un número absurdamente alto de filas."""
    assert len(raw_data) < 10000


def test_validar_calidad_datos_pass(raw_data):
    """Verifica que validar_calidad_datos retorna PASS para datos válidos."""
    reporte = validar_calidad_datos(raw_data, COLS_GASTO)
    assert all(r["pass"] for r in reporte.values()), (
        f"Reglas fallidas: {[k for k, v in reporte.items() if not v['pass']]}"
    )
