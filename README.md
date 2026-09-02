# Grupo 6 — MLOps End-to-End: Segmentación de Clientes Mayoristas (Wholesale Customers)

> Curso: MLOps End-to-End | Proyecto (60%)
> Tipo de problema: **Clustering (no supervisado)**
> Dataset: [Wholesale Customers — UCI ML Repository (ID 292)](https://archive.ics.uci.edu/dataset/292/wholesale+customers)

---

## 1. Business Problem

El dataset representa el gasto anual (en unidades monetarias) de clientes de
un distribuidor mayorista, distribuido en 6 categorías de producto (Fresh,
Milk, Grocery, Frozen, Detergents_Paper, Delicassen).

El objetivo del proyecto es **descubrir estructuras/segmentos de clientes
nuevos** a partir únicamente del comportamiento de gasto, **sin usar
inicialmente** las columnas `Channel` (canal Horeca/Retail) ni `Region`
como variables de entrenamiento. Estas dos columnas se usan únicamente
al final, como validación externa, para verificar si los segmentos
descubiertos capturan información más rica que la ya conocida.

## 2. Dataset

- **Fuente:** UCI Machine Learning Repository, dataset ID 292.
- **Tamaño:** 440 registros, 8 columnas, sin valores faltantes.
- **Licencia:** CC BY 4.0.
- **Nota importante:** el CSV crudo **no se sube directamente** como fuente
  de verdad del proyecto (sección D del enunciado). Se obtiene mediante
  `src/ingestion/ingest.py`. Una copia pequeña de respaldo
  (`data/raw/_fallback/`) se mantiene en el repo únicamente como plan de
  contingencia para demos sin conexión a internet — ver sección 6.

## 3. Architecture

Arquitectura general obligatoria del curso:

```
Fuente de datos → Data Ingestion → Raw/Bronze → Data Validation
   → (FAIL → Alert | PASS → Data Cleaning) → Feature Pipeline
   → Training → Evaluation → MLflow (Tracking + Registry) → Best Candidate
   → Dockerize → Model API → Production → Monitoring
   (Data Drift / Model Performance / System Metrics) → Retrain Trigger
```

## 4. Repository Structure

Estructura actual:

```
grupo6-wholesale-mlops/
├── README.md
├── requirements.txt
├── .gitignore
├── .dockerignore
├── Dockerfile
├── mlflow.db # generado por train.py/notebooks (no versionado)
│
├── data/
│ ├── raw/
│ │ ├── _fallback/ # copia pequeña de respaldo (offline demo)
│ │ ├── wholesale_customers_raw.csv # generado por ingest.py (no versionado)
│ │ └── ingestion_metadata.json # generado por ingest.py (no versionado)
│ └── processed/
│ ├── wholesale_features.csv # generado por build_features.py (no versionado)
│ └── features_metadata.json # generado por build_features.py (no versionado)
│
├── docs/
│ ├── architecture.mmd # diagrama de arquitectura (fuente Mermaid)
│ ├── architecture.png # diagrama de arquitectura (imagen)
│ └── informe_tecnico.md # informe técnico del proyecto
│
├── logs/
│ └── quality_alerts.log # generado por emitir_alerta() (no versionado)
│
├── models/
│ ├── kmeans_production.joblib # generado por train.py en promoción a Production (no versionado)
│ ├── feature_builder.joblib # generado por build_features.py / train.py (no versionado)
│ └── production_metadata.json # generado por train.py en promoción a Production (no versionado)
│
├── notebooks/
│ ├── 01_data_quality_diagnostico.ipynb # ✅ implementado (sección F)
│ ├── 02_eda_feature_engineering.ipynb # ✅ implementado (secciones H, I)
│ ├── 03_modelado_experiment_tracking.ipynb # ✅ implementado (secciones J, K)
│ └── 04_monitoring.ipynb # ✅ implementado (secciones O, P, Q, R)
│
├── src/
│ ├── ingestion/
│ │ └── ingest.py # ✅ implementado
│ ├── data_quality/
│ │ └── validate.py # ✅ implementado (7 reglas + alertas persistentes)
│ ├── features/
│ │ └── build_features.py # ✅ implementado (FeatureBuilder)
│ ├── training/
│ │ └── train.py # ✅ implementado (secciones J, K)
│ ├── monitoring/ # ✅ implementado (secciones O, R)
│ │ ├── system_metrics.py # O1: latency, throughput, error rate, availability
│ │ ├── drift.py # O2: PSI (Population Stability Index)
│ │ ├── model_monitor.py # O3: distribución y estabilidad de clusters
│ │ ├── retrain_trigger.py # R: lógica de decisión de reentrenamiento
│ │ └── run_monitoring.py # Script ejecutable de monitoreo (O1+O2+O3+R)
│ └── api/
│ ├── main.py # ✅ implementado (FastAPI + middleware métricas)
│ └── schemas.py # ✅ implementado (Pydantic)
│
└── tests/
├── conftest.py # ✅ implementado (fixtures)
├── test_ingestion.py # ✅ implementado (sección N)
├── test_data.py # ✅ implementado (sección N)
├── test_model.py # ✅ implementado (sección N)
├── test_api.py # ✅ implementado (sección N)
└── test_monitoring.py # ✅ implementado (sección O monitoring)

## 5. Installation

```bash
git clone https://github.com/<usuario>/grupo6-wholesale-mlops.git
cd grupo6-wholesale-mlops

python -m venv venv        # o: py -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 6. Data Ingestion

Script reproducible: `src/ingestion/ingest.py`

```bash
python src/ingestion/ingest.py     # o: py src/ingestion/ingest.py
```

**Estrategia de ingesta (en orden de intento):**

1. Paquete oficial `ucimlrepo` (`fetch_ucirepo(id=292)`).
2. Descarga directa del CSV publicado por UCI vía URL.
3. Copia local de respaldo (`data/raw/_fallback/`) — **solo** si no hay
   conexión a internet disponible. El script emite un `WARNING` explícito
   cuando usa este método, para dejar evidencia clara de que fue la
   excepción y no la fuente primaria.

**Salidas del script:**

- `data/raw/wholesale_customers_raw.csv` — dataset crudo ingerido (no se
  versiona en git, se regenera en cada ejecución).
- `data/raw/ingestion_metadata.json` — metadata de trazabilidad: método de
  ingesta usado, timestamp UTC, checksum SHA-256, número de filas/columnas
  y un `data_version` (usado luego como parámetro en MLflow).

**Validación mínima incluida:** el script verifica que las 8 columnas
esperadas existan y que el dataset tenga al menos 400 filas antes de
aceptar la ingesta como válida. Esta es la primera capa de validación;
las reglas de calidad más profundas se implementan en
`src/data_quality/validate.py` (documentado en la siguiente sección).

## 7. Data Quality

**Diagnóstico exploratorio:** `notebooks/01_data_quality_diagnostico.ipynb`

El notebook investiga, con evidencia y justificación para cada decisión
(no transformaciones aplicadas "a ciegas"):

- Valores faltantes explícitos y codificados (ej. ceros/negativos sospechosos)
- Duplicados totales y duplicados parciales por perfil de gasto
- Cardinalidad y validez de `Channel` / `Region` frente a lo documentado en UCI
- Valores extremos y datos imposibles (gastos negativos)
- Asimetría y correlación excesiva entre variables de gasto

**Hallazgos principales:**

- El dataset **no presenta** nulos, duplicados, ni valores negativos.
- Los valores extremos detectados corresponden a clientes reales de alto
  volumen de compra, no a errores de captura — por lo tanto **no se
  eliminan**, se tratan con transformación logarítmica en la etapa de
  Feature Engineering (ver sección 8).
- Las seis variables de gasto presentan **asimetría alto** (todas por
  encima de 2 en valor absoluto, `Delicassen` la más extrema con 11.15),
  lo que confirma la necesidad de transformar antes de aplicar algoritmos
  de clustering sensibles a escala/distribución como K-Means.

**Data Quality Gates (reglas automáticas):** `src/data_quality/validate.py`

```bash
python src/data_quality/validate.py     # o: py src/data_quality/validate.py
```

Implementa 7 reglas automáticas, cada una con umbral justificado en el
notebook de diagnóstico:

| Regla                            | Qué valida                                                                | Umbral                     |
| -------------------------------- | ------------------------------------------------------------------------- | -------------------------- |
| `min_rows`                       | El dataset no perdió filas significativamente respecto al histórico (440) | ≥ 90% del histórico        |
| `sin_nulos_obligatorios`         | No hay nulos en columnas de gasto ni en Channel/Region                    | 0 nulos                    |
| `duplicados_bajo_umbral`         | Proporción de filas duplicadas                                            | < 2%                       |
| `tipos_numericos_validos`        | Columnas de gasto son numéricas                                           | Todas numéricas            |
| `sin_gastos_negativos`           | Ninguna variable de gasto tiene valores negativos (dato imposible)        | 0 negativos                |
| `cardinalidad_categorica_valida` | Channel ∈ {1,2} y Region ∈ {1,2,3}                                        | sin categorías inesperadas |
| `esquema_sin_columnas_extra`     | No hay columnas fuera del esquema esperado (8 columnas)                   | 0 columnas extra           |

Este script está diseñado para ejecutarse tanto de forma independiente
como importado desde otros scripts del pipeline (`src/training/train.py`),
para que la validación ocurra automáticamente antes de entrenar, sin
depender de que alguien la corra manualmente desde el notebook.

## 8. EDA y Feature Engineering

**Análisis exploratorio orientado a decisiones:**
`notebooks/02_eda_feature_engineering.ipynb`

Cada análisis de este notebook responde explícitamente la pregunta que
exige la sección H del enunciado: _¿qué decisión de modelado, limpieza,
ingeniería de variables o negocio cambia como consecuencia de este
resultado?_ Los hallazgos principales:

| Hallazgo                                                                                                                    | Decisión                                                                                                       | Se implementa en                                                            |
| --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Asimetría alto en las 6 variables de gasto (todas > 2, `Delicassen` = 11.15)                                                | Log-transform (log1p)                                                                                          | `FeatureBuilder._log_transform`                                             |
| Correlación alta entre Grocery, Detergents_Paper y Milk (hasta 0.92) + colinealidad perfecta en las proporciones (Σpct = 1) | PCA con k elegido por varianza acumulada ≥80% (k=5, retiene 85.7%)                                             | `FeatureBuilder(n_pca_components=5)`                                        |
| Outliers son clientes reales de alto volumen (`Frozen` el más frecuente, 43 casos), no errores                              | No se eliminan; se comprimen con log-transform                                                                 | `FeatureBuilder._log_transform`                                             |
| Colas ya acotadas tras log1p (máx. \|z\| = 6.36 vs 4.54 con RobustScaler)                                                   | StandardScaler; RobustScaler no es necesario                                                                   | `FeatureBuilder.scaler`                                                     |
| El perfil de composición del gasto es distinto al volumen total                                                             | Proporciones por categoría + ratio perecederos/no perecederos + índice de diversificación (entropía)           | `_proporciones_gasto`, `_ratio_perecederos`, `_indice_diversificacion`      |
| Preview de clusterabilidad: silhouette más alto en k=2 (0.322), posible riesgo de "redescubrir" Channel                     | Confirmado en training (sección 9): k=3 elegido por aportar valor de negocio adicional pese a menor silhouette | `src/training/train.py` y `notebooks/03_modelado_experiment_tracking.ipynb` |
| El enfoque del proyecto pide descubrir estructura sin partir de categorías ya conocidas                                     | No usar Channel/Region como input del clustering; solo validación externa posterior                            | Validación externa hecha en el notebook de modelado (sección 9)             |

**Feature Engineering reutilizable:** `src/features/build_features.py`

```bash
python src/features/build_features.py     # o: py src/features/build_features.py
```

La clase `FeatureBuilder` es la **única fuente de verdad** de las
transformaciones del proyecto — se importa igual desde los notebooks de
EDA/modelado y desde `src/training/train.py`, evitando el antipatrón
de tener una lógica de features en el notebook y otra distinta en
producción (sección I del enunciado). Provee:

- Transformaciones: log-transform, proporciones de gasto, ratio
  perecederos/no perecederos, índice de diversificación, escalado y PCA.
- Interfaz `fit_transform` / `transform` estilo scikit-learn, para nunca
  reajustar el scaler/PCA con datos nuevos en producción.
- Persistencia (`save` / `load` vía `joblib`) del builder ya ajustado, para
  que la futura API de inferencia aplique exactamente las mismas
  transformaciones sobre datos nuevos.

**Salidas del script:**

- `data/processed/wholesale_features.csv` — dataset de features listo para
  entrenamiento (no se versiona, se regenera con el script).
- `data/processed/features_metadata.json` — trazabilidad: configuración de
  features usada, checksums, y el `data_version` heredado de la ingesta
  (parámetro que se registrará en MLflow).
- `models/feature_builder.joblib` — el `FeatureBuilder` ajustado y
  serializado, reutilizable en entrenamiento e inferencia.

## 9. Training

**Exploración y comparación de modelos:**
`notebooks/03_modelado_experiment_tracking.ipynb`

Se compararon **10 configuraciones** (2 algoritmos × k=2 a 6): K-Means y
Clustering Jerárquico (Agglomerative). K-Means dominó consistentemente en
silhouette y Davies-Bouldin en los 6 valores de k probados, por lo que la
comparación final de candidatos se hizo entre **k=2** y **k=3** con K-Means.

**Criterio explícito de selección (sección K):**

| Candidato                 | Silhouette              | Davies-Bouldin | Observación clave                                                                                                                                                           |
| ------------------------- | ----------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| K-Means k=2               | **0.322** (el más alto) | 1.231          | Cluster dominante es 96% Channel=Horeca — redescubre casi directamente la variable que el proyecto decidió no usar como input                                               |
| **K-Means k=3 (elegido)** | 0.257 (segundo mejor)   | 1.393          | Introduce un tercer perfil de negocio genuinamente distinto: separa "Horeca fresco-especializado" de "Horeca diversificado", información que Channel por sí solo no captura |

**Se eligió k=3** sobre k=2 pese a su silhouette más bajo, porque aporta
segmentación de negocio más rica y no se limita a redescubrir una variable
ya conocida — cumpliendo el enfoque original del proyecto (sección
"Business Problem"). Ambos candidatos tienen clusters balanceados (sin
grupos degenerados) y fueron validados cruzando contra `Channel`/`Region`
como verificación externa, nunca como input del modelo.

**Validación de estabilidad:** antes de promover el modelo, se re-entrenó
k=3 con una semilla alternativa (123). El silhouette varió de 0.257 a
0.256 (diferencia de 0.000, muy por debajo del umbral de 0.05 definido),
confirmando que el resultado es estable y no depende de la inicialización
aleatoria.

**Reproducción en producción:** `src/training/train.py`

```bash
python src/training/train.py     # o: py src/training/train.py
```

Reproduce en código de producción la configuración ganadora decidida en
el notebook (K-Means, k=3, `K_PCA=5`), sin reimplementar la lógica de
features (importa `FeatureBuilder`) ni la de calidad de datos (importa
`validar_calidad_datos`). Cada ejecución:

1. Carga los datos crudos y valida las Data Quality Gates (detiene el
   pipeline si alguna falla).
2. Construye las features con `FeatureBuilder`.
3. Entrena K-Means (k=3) y calcula silhouette, Davies-Bouldin e inertia.
4. Registra un nuevo run en MLflow con parámetros, métricas, el modelo
   como artifact y un scatter plot de clusters.
5. Valida estabilidad con una semilla alternativa y, si pasa, promueve el
   modelo a Production en el Model Registry.

## 10. MLflow

**Tracking URI:** SQLite (`mlflow.db`, en la raíz del repo) — se eligió
sobre el file store por defecto porque el Model Registry (sección K)
requiere un backend con soporte completo de versiones de modelo.

**Ver la interfaz de MLflow:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Cada run registra (mínimo exigido por la sección J):**

| Categoría      | Contenido registrado                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Parameters** | `algorithm`, `n_clusters`, `feature_set` (ej. `PCA_5_componentes`), `random_seed`, `data_version` (heredado de `ingestion_metadata.json`)        |
| **Metrics**    | `silhouette`, `davies_bouldin`, `inertia`; en el run final también `silhouette_semilla_alternativa` y `diferencia_estabilidad`                   |
| **Artifacts**  | El modelo entrenado (`mlflow.sklearn.log_model`), scatter plot de clusters (desde `train.py`), y gráficos de comparación de clusters y perfiles de gasto generados en el notebook |

**Model Registry — ciclo Experiment → Candidate → Validation → Production:**

1. **Experiment:** los 10 runs de comparación (k=2-6 × 2 algoritmos) quedan
   registrados bajo el experimento `wholesale-clustering-grupo6`.
2. **Candidate:** el run ganador (K-Means, k=3) se registra con
   `mlflow.register_model()` y se etiqueta `lifecycle_stage=candidate`.
3. **Validation:** se re-entrena con una semilla alternativa; si la
   diferencia de silhouette es menor al umbral (0.05), se etiqueta
   `lifecycle_stage=validation`.
4. **Production:** se le asigna el alias `production` con
   `set_registered_model_alias()` (API moderna de MLflow — se evitó
   `transition_model_version_stage()` por estar deprecada desde la
   versión 2.9.0), garantizando que este mecanismo sea idéntico tanto en
   el notebook de exploración como en `src/training/train.py`.

Este ciclo responde directamente la pregunta que exige la sección J:
_¿exactamente qué datos, código, features, hiperparámetros y métricas
produjeron este modelo?_ — toda esa información queda trazada en el run
registrado bajo el alias `production`.

## 11. Docker

El modelo se sirve dentro de un contenedor Docker ligero (sección L del
enunciado). La imagen usa `python:3.11-slim` (~150 MB vs ~900 MB de la
imagen completa) e instala **solo** las dependencias necesarias para la
API de inferencia (sin seaborn, matplotlib, jupyter, etc.).

**Construir la imagen:**

```bash
docker build -t grupo6-mlops .
```

**Ejecutar el contenedor:**

```bash
# Lo mínimo necesario (incluye los artefactos del modelo en la imagen)
docker run -p 8000:8000 grupo6-mlops

# Opcional: montar models/ como volumen para no reconstruir la imagen
# cuando se re-entrene el modelo
docker run -p 8000:8000 -v "${PWD}/models:/app/models" grupo6-mlops
```

**Verificar que funciona:**

```powershell
# PowerShell
Invoke-RestMethod -Uri http://localhost:8000/health
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body '{"Fresh": 12669, "Milk": 9656, "Grocery": 7561, "Frozen": 214, "Detergents_Paper": 2674, "Delicassen": 1338}'
```

**Dockerfile:** incluye un `HEALTHCHECK` que consulta `/health`
automáticamente cada 30s. Las dependencias están fijadas con versiones
exactas (no rangos) para garantizar reproduciibilidad — evita el
problema "en mi computadora sí funciona" por incompatibilidad de
versiones de scikit-learn al deserializar el modelo con joblib.

**`.dockerignore`:** excluye `venv/`, `data/`, `mlflow.db`, `mlruns/`,
`mlartifacts/`, `notebooks/`, `.git/` — nada de esto entra a la imagen.

## 12. API de inferencia

API REST construida con **FastAPI** que sirve predicciones de clustering
(sección M del enunciado). Carga el modelo K-Means y el FeatureBuilder
desde archivos joblib exportados por `train.py` — **no depende de
MLflow ni de la base de datos** en tiempo de inferencia.

**Código fuente:** `src/api/main.py` + `src/api/schemas.py`

**Levantar localmente (sin Docker):**

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Documentación interactiva (Swagger):** `http://localhost:8000/docs`

**Probar la API (PowerShell):**

```powershell
# Verificar salud
Invoke-RestMethod -Uri http://localhost:8000/health

# Predecir cluster para un cliente
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body '{"Fresh": 12669, "Milk": 9656, "Grocery": 7561, "Frozen": 214, "Detergents_Paper": 2674, "Delicassen": 1338}'
```

### Endpoints

| Método | Ruta       | Descripción                                      |
| ------ | ---------- | ------------------------------------------------ |
| `GET`  | `/`        | Información básica de la API                     |
| `GET`  | `/health`  | Estado del servicio y versión del modelo cargado |
| `GET`  | `/metrics` | Métricas de sistema: latencia, throughput, error rate, availability (Sección O1) |
| `POST` | `/predict` | Predicción de cluster para un cliente nuevo      |

### Formato de request (POST /predict)

El endpoint acepta las 6 categorías de gasto del dataset original.
`Channel` y `Region` **no se piden** a propósito — el modelo fue
entrenado sin usarlas como input.

```json
{
  "Fresh": 12669,
  "Milk": 9656,
  "Grocery": 7561,
  "Frozen": 214,
  "Detergents_Paper": 2674,
  "Delicassen": 1338
}
```

### Formato de respuesta

```json
{
  "cluster": 1,
  "distance_to_centroid": 2.1244,
  "model_version": "5"
}
```

Los campos `Channel` y `Region` se completan internamente como `NaN`
para satisfacer la forma esperada por `FeatureBuilder` — no afectan el
cálculo del cluster ni la distancia.

### Artefactos cargados al iniciar

| Archivo                           | Origen                                 | Propósito                                 |
| --------------------------------- | -------------------------------------- | ----------------------------------------- |
| `models/kmeans_production.joblib` | `train.py` (en promoción a Production) | Modelo K-Means entrenado                  |
| `models/feature_builder.joblib`   | `train.py` (en promoción a Production) | Transformaciones ajustadas (scaler + PCA) |
| `models/production_metadata.json` | `train.py` (en promoción a Production) | Versión, métricas, run_id de MLflow       |

**Error si los artefactos no existen:** la API levanta pero retorna
HTTP 503 en `/predict` hasta que se ejecuten `python src/training/train.py`
para generar los modelos.

**Ejecutar dentro de Docker:** ver sección 11 (Docker).

## 13. Pruebas

Suite de tests con **pytest** que cubre datos, modelo, API y
monitoring. Total: 46 tests incondicionales (+ 5 condicionales que requieren modelos generados).

```bash
pytest tests/ -v
```

### Datos (`tests/test_data.py`)

Verifican que el dataset cumple las expectativas del esquema original:

| Test | Qué valida |
|------|------------|
| `test_esquema_columnas` | Las 8 columnas esperadas existen |
| `test_tipos_numericos` | Todas las columnas son numéricas |
| `test_rango_channel` | Channel ∈ {1, 2} |
| `test_rango_region` | Region ∈ {1, 2, 3} |
| `test_sin_nulos` | 0 valores faltantes |
| `test_sin_gastos_negativos` | Ninguna variable de gasto es negativa |
| `test_minimo_400_filas` | Al menos 400 registros |

### Modelo (`tests/test_model.py`)

Verifican que el modelo entrenado funciona correctamente:

| Test | Qué valida |
|------|------------|
| `test_modelo_carga` | El modelo KMeans se carga de `models/kmeans_production.joblib` |
| `test_feature_builder_carga` | El FeatureBuilder se carga y está ajustado |
| `test_input_valido_genera_prediccion` | Input válido → cluster ∈ {0, 1, 2} |

### API (`tests/test_api.py`)

Verifican los endpoints con `FastAPI TestClient`:

| Test | Qué valida |
|------|------------|
| `test_health_200` | GET /health → 200 + `status: "ok"` |
| `test_metrics_200` | GET /metrics → 200 + 6 claves de métricas |
| `test_root_200` | GET / → 200 |
| `test_predict_200` | Request válido → HTTP 200 + schema válido |
| `test_predict_campos_requeridos` | Campo faltante → HTTP 422 |
| `test_predict_gasto_negativo` | Gasto negativo → HTTP 422 |

**Prerequisito:** los tests de modelo y API requieren que existan los
artefactos en `models/` y el CSV en `data/raw/`. Ejecutar primero
`python src/training/train.py` si no existen.

### Monitoring (`tests/test_monitoring.py`)

Verifican la lógica de los módulos de monitoreo (drift, model_monitor,
system_metrics, retrain_trigger):

| Test | Qué valida |
|------|------------|
| `test_clasificar_psi_ok` | PSI < 0.10 → "OK" |
| `test_clasificar_psi_warning` | 0.10 ≤ PSI < 0.25 → "WARNING" |
| `test_clasificar_psi_alert` | PSI ≥ 0.25 → "ALERT" |
| `test_calcular_psi_mismas_distribuciones` | Distribuciones idénticas → PSI ≈ 0 |
| `test_calcular_psi_distribuciones_diferentes` | Distribuciones distintas → PSI > 0 |
| `test_comparar_distribuciones_estable` | Diferencia < 10pp → estable |
| `test_comparar_distribuciones_inestable` | Diferencia > 10pp → inestable |
| `test_get_metrics_estructura` | Retorna las 6 claves de métricas de sistema |
| `test_cargar_modelo` | Modelo y FeatureBuilder se cargan correctamente |
| `test_obtener_distribucion_clusters` | Distribución suma ~100%, clusters ∈ {0,1,2} |
| `test_evaluar_ok` | Drift bajo + silhouette alta + composición estable → "OK" |
| `test_evaluar_monitorear` | Drift alto pero modelo estable → "MONITOREAR" |
| `test_evaluar_reentrenar` | Composición inestable → "REENTRENAR" |

## 14. Monitoring (Secciones O, P, Q, R)

El monitoreo es uno de los componentes de mayor peso del proyecto. Se
implementa en `src/monitoring/` y se demuestra en
`notebooks/04_monitoring.ipynb`. Cubre tres dimensiones (O1–O3), una
simulación de drift (P), una simulación de contaminación de calidad (Q)
y una estrategia de reentrenamiento (R).

### 14.1 O1 — System Monitoring

**Módulo:** `src/monitoring/system_metrics.py`

Métricas operativas de corto plazo, calculadas en memoria del proceso:

| Métrica                  | Descripción                                             |
| ------------------------ | ------------------------------------------------------- |
| `latency_avg_ms`         | Latencia promedio de las últimas 1000 requests (ventana móvil) |
| `throughput_req_per_sec` | Requests por segundo desde el arranque del servicio     |
| `error_rate_pct`         | Porcentaje de requests con status ≥ 400                 |
| `availability_pct`       | Porcentaje de requests exitosos                         |
| `total_requests`         | Total acumulado de requests                             |
| `uptime_seconds`         | Tiempo desde el arranque del servicio                   |

**Endpoint:** `GET /metrics` — expone las métricas en JSON.

**Middleware:** `src/api/main.py` incluye un middleware HTTP que mide
la latencia de cada request y alimenta `system_metrics.py` vía
`record_request()`.

**Limitación conocida:** el estado es por proceso (in-memory). Con
múltiples workers de uvicorn, cada worker tendría su propio estado.
Para este proyecto (un solo worker vía Docker) no es un problema.

**Evidencia:** el notebook 04 demuestra las métricas tras ejecutar
los batches de producción simulados.

### 14.2 O2 — Data Monitoring (Drift)

**Módulo:** `src/monitoring/drift.py`

Detecta cambios en la distribución de las variables de entrada
comparando `P_reference(X)` contra `P_production(X)` mediante
**PSI (Population Stability Index)**.

**Umbrales** (justificados en `notebooks/04_monitoring.ipynb`, no
tratados como leyes universales):

| PSI              | Clasificación |
| ---------------- | ------------- |
| `< 0.10`         | OK            |
| `≥ 0.10, < 0.25` | WARNING       |
| `≥ 0.25`         | ALERT         |

**Funciones principales:**

- `calcular_psi(referencia, produccion, bins=10)` — PSI con
  suavizado de Laplace (evita división por cero y log(0)).
- `calcular_psi_dataframe(df_ref, df_prod, columnas)` — PSI por
  columna para las 6 variables de gasto.
- `clasificar_psi(valor_psi)` — clasifica PSI según los umbrales.

### 14.3 O3 — Model Monitoring

**Módulo:** `src/monitoring/model_monitor.py`

Para el problema de clustering, monitorea la **distribución de
clusters** y la **estabilidad** entre un batch de referencia y uno
de producción.

**Funciones principales:**

- `obtener_distribucion_clusters(modelo, feature_builder, df)` —
  predice clusters y retorna el % de registros en cada uno.
- `comparar_distribuciones(dist_ref, dist_prod, umbral=10.0)` —
  compara dos distribuciones. Si algún cluster gana o pierde más de
  **10 puntos porcentuales** de participación, se marca como
  inestable.

**Justificación del umbral de 10pp:** los 3 clusters del modelo
están relativamente balanceados (30–38% cada uno, ver notebook 03).
Un desplazamiento de 10pp movería a algún cluster fuera de ese rango
típico.

**Nota sobre desplazamiento de centroides:** no se trackea porque
KMeans no se reentrena en cada batch de producción — los centroides
solo cambian cuando corre `train.py`, momento que queda trazado en
MLflow (Sección J).

### 14.4 P — Simulación de Producción y Drift

**Notebook:** `notebooks/04_monitoring.ipynb` (Secciones P y O2/O3)

Se divide conceptualmente el dataset en 4 partes:

```
REFERENCE → BATCH 1 (sin drift) → BATCH 2 (drift leve) → BATCH 3 (drift fuerte)
```

| Batch      | Diseño                                              | PSI_max | Clasificación | Modelo estable |
| ---------- | --------------------------------------------------- | ------- | ------------- | -------------- |
| BATCH 1    | Bootstrap de REFERENCE (muestra aleatoria)          | 0.042   | OK            | Sí (4.01pp)    |
| BATCH 2    | +20–40% en Fresh/Milk/Grocery, ruido alto          | 0.131   | WARNING       | Sí (3.35pp)    |
| BATCH 3    | +120% en Fresh/Frozen/Delicassen, ruido bajo       | 0.571   | ALERT         | No (15.73pp)   |

**Hallazgo clave (Sección O3):** BATCH 2 tiene drift real
(PSI=0.131, WARNING) pero el modelo se mantiene estable — el drift
no se traduce en degradación. BATCH 3 sí desestabiliza el modelo:
el cluster 1 ("retail/abarrotes") pierde casi 16pp de participación
hacia el cluster 2 ("Horeca diversificado"). Esta progresión
sustenta el criterio de reentrenamiento de la Sección R.

### 14.5 Q — Simulación de Problemas de Calidad

**Notebook:** `notebooks/04_monitoring.ipynb` (Sección Q)

Se inyectan los 6 tipos de defecto que exige el enunciado sobre una
copia del dataset de referencia:

| Defecto inyectado                      | Regla que lo detecta                 | Resultado         |
| -------------------------------------- | ------------------------------------ | ----------------- |
| Missing values (Fresh, Milk)           | `sin_nulos_obligatorios`             | FAIL — detectado  |
| Fila duplicada                         | `duplicados_bajo_umbral`             | PASS (0.23% < 2%) |
| Outlier extremo (Frozen = -500000)     | `sin_gastos_negativos`               | FAIL — detectado  |
| Tipo incorrecto (texto en numérico)    | `tipos_numericos_validos`            | FAIL — detectado  |
| Categoría desconocida (Channel = 99)   | `cardinalidad_categorica_valida`     | FAIL — detectado  |
| Columna extra (columna_falsa)          | `esquema_sin_columnas_extra`         | FAIL — detectado  |

**Ciclo validado:** Detecta → Bloquea/Advierte → Registra. El batch
contaminado se descarta (`del batch_contaminado`); el dataset original
nunca se modifica (REFERENCE se mantuvo en 440 filas tras la prueba).

El caso del duplicado (PASS con 0.23%) no es un fallo del sistema: el
umbral de 2% se justificó con evidencia real (0% de duplicados en el
histórico) y el detalle del reporte lo refleja explícitamente — el
sistema "vio" el duplicado pero no lo consideró suficiente para
bloquear por sí solo.

### 14.6 R — Estrategia de Reentrenamiento

**Módulo:** `src/monitoring/retrain_trigger.py`

El proyecto no implementa un sistema autónomo completo de Continuous
Training, pero sí la **lógica de decisión** que determinaría cuándo
reentrenar. La función `evaluar_necesidad_reentrenamiento()` combina
tres señales:

| Señal                          | Cómo se mide                                       | Umbral                |
| ------------------------------ | -------------------------------------------------- | --------------------- |
| **Drift de datos**             | PSI máximo entre las 6 variables de gasto          | > 0.25 (ALERT)        |
| **Degradación de silhouette**  | Silhouette actual vs. línea base (production_metadata.json) | < 70% de la línea base |
| **Inestabilidad de composición** | Desplazamiento máximo entre distribuciones de clusters | > 10pp                |

**Decisión:**

| Condición                                    | Decisión      |
| -------------------------------------------- | ------------- |
| Drift alto pero modelo estable               | MONITOREAR    |
| Silhouette bajo O composición inestable      | REENTRENAR    |
| Ninguna señal                                | OK            |

**¿Por qué drift solo NO implica reentrenar?**

BATCH 2 demuestra que drift real (PSI=0.131, WARNING) no se traduce
en degradación del modelo (composición estable, silhouette sin caída
significativa). Reentrenar en cada cambio de distribución
desperdiciaría cómputo y arriesgaría introducir inestabilidad. Un
cambio de distribución puede ser legítimo (estacionalidad, promoción
puntual) y no dañino — solo se actúa cuando hay evidencia de
degradación real.

**Resultado de la simulación (notebook 04, Sección R):**

| Batch    | PSI    | Silhouette | Composición | Decisión      |
| -------- | ------ | ---------- | ----------- | ------------- |
| BATCH 1  | 0.042  | 0.2575     | Estable     | OK            |
| BATCH 2  | 0.131  | 0.2535     | Estable     | OK            |
| BATCH 3  | 0.571  | 0.2083     | Inestable   | REENTRENAR    |

BATCH 3 dispara REENTRENAR, pero no por caída de silhouette
(0.2083, todavía por encima del umbral 0.1796), sino por
inestabilidad de composición de clusters (15.73pp vs. umbral de
10pp). Esto confirma que la tercera señal (composición) es necesaria
y no redundante con el silhouette.

## 15. Results

### 15.1 Métricas del Modelo en Producción

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Silhouette Score** | 0.2566 | Separación moderada entre clusters |
| **Davies-Bouldin Index** | 1.4325 | Índice aceptable (menor = mejor) |
| **Inertia** | 2841.48 | Cohesión intra-cluster |
| **Estabilidad (diff semilla)** | 0.0002 | Muy estable (umbral: 0.05) |

### 15.2 Comparación de Configuraciones

Se evaluaron 10 configuraciones (KMeans y Agglomerative Clustering, k=2 a k=6). KMeans domina en todas las métricas para cada valor de k.

| Algoritmo | k | Silhouette | Davies-Bouldin |
|-----------|---|------------|----------------|
| KMeans | 2 | 0.3224 | 1.2315 |
| **KMeans** | **3** | **0.2566** | **1.4325** |
| KMeans | 4 | 0.2320 | 1.4155 |
| KMeans | 5 | 0.2344 | 1.3935 |
| KMeans | 6 | 0.2334 | 1.3506 |

**Selección de k=3:** Aunque k=2 tenía el mejor silhouette (0.322), su cluster dominante era 96% Channel=Horeca — básicamente redescubría una variable conocida. k=3 revela un perfil de negocio genuinamente nuevo (HoreCa diversificado) sin ser redundante con Channel.

### 15.3 Perfiles de Clusters (Datos Originales, Promedio)

| Cluster | Fresh | Milk | Grocery | Frozen | Detergents_Paper | Delicassen | Perfil |
|---------|-------|------|---------|--------|------------------|------------|--------|
| **0** | **19,515** | 2,057 | 2,632 | 3,030 | 420 | 763 | HoReCa Fresh |
| **1** | 6,221 | **9,596** | **14,870** | 1,266 | **6,550** | 1,283 | Retail/Abarrotes |
| **2** | 11,604 | 4,890 | 4,855 | **5,269** | 891 | **2,554** | HoReCa Diversificado |

**Interpretación de negocio:**
- **Cluster 0 (HoReCa Fresh — 30.7%):** Clientes con gasto extremadamente alto en Fresh (19,515) y bajo en todo lo demás. Restaurantes y hoteles especializados en alimentos frescos.
- **Cluster 1 (Retail/Abarrotes — 37.7%):** Dominado por Milk (9,596), Grocery (14,870) y Detergents_Paper (6,550). Tiendas de abarrotes y supermercados.
- **Cluster 2 (HoReCa Diversificado — 31.6%):** Fresh moderado, pero el gasto más alto en Frozen (5,269) y Delicassen (2,554). Restaurantes con menú diversificado que incluye congelados y delicatessen.

### 15.4 Validación Externas

**Cluster vs Channel:**

| Cluster | Channel=1 (Horeca) | Channel=2 (Retail) |
|---------|-------------------|-------------------|
| 0 | **97%** | 3% |
| 1 | 27% | **73%** |
| 2 | **88%** | 12% |

Los clusters 0 y 2 son ambos dominados por Horeca, pero representan dos perfiles de compra genuinamente distintos que Channel solo no captura: el especializado en fresco vs. el diversificado.

**Cluster vs Region:** Sin poder discriminatorio — todos los clusters son ~70% Region 3.

### 15.5 Análisis de Componentes Principales

5 componentes retienen el **85.7%** de la varianza:

| Componente | Varianza | Acumulada |
|------------|----------|-----------|
| PC1 | 41.8% | 41.8% |
| PC2 | 16.3% | 58.2% |
| PC3 | 12.1% | 70.3% |
| PC4 | 9.2% | 79.5% |
| PC5 | 6.2% | **85.7%** |

## 16. Team

| Integrantes           |
| --------------------- |
| Ismael Quesada Salas  |
| Magaly Bushey Ventura |
