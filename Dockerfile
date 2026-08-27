# Dockerfile — Wholesale Customers Clustering API (Grupo 6)
#
# Imagen ligera basada en Python slim (no la imagen completa de Python,
# que pesa varias veces más y no la necesitamos para servir un modelo
# ya entrenado). Esto atiende directamente el criterio de "tamaño
# razonable" de la sección L del enunciado.

FROM python:3.11-slim

# Evita que pip guarde caché de descargas dentro de la imagen (reduce tamaño)
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copiamos solo requirements.txt primero (no todo el código) para
# aprovechar el cache de capas de Docker: si el código cambia pero las
# dependencias no, Docker no reinstala todo desde cero en cada build.
COPY requirements.txt .

# Instalamos solo lo necesario para SERVIR la API, no todo lo del
# proyecto de entrenamiento/EDA (evita instalar seaborn, matplotlib,
# jupyter, etc. dentro del contenedor de producción). Versiones FIJAS
# (no rangos) para garantizar que el contenedor use exactamente las
# mismas versiones con las que train.py generó el modelo — evita el
# problema de "en mi computadora sí funciona" por incompatibilidad de
# versiones de scikit-learn al deserializar el modelo con joblib.
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.4.0 \
    scikit-learn==1.8.0 \
    pandas==2.3.3 \
    numpy==2.4.4 \
    joblib==1.5.3

# Copiamos únicamente lo que la API necesita para funcionar en producción:
# el código de la API, el módulo de features (FeatureBuilder es requerido
# para transformar inputs nuevos), y los artefactos del modelo ya
# entrenado. NO copiamos notebooks, datos crudos, ni el código de
# entrenamiento — eso vive fuera del contenedor de serving.
COPY src/api/ ./src/api/
COPY src/features/ ./src/features/
COPY models/ ./models/

# FastAPI/uvicorn escuchará en este puerto dentro del contenedor
EXPOSE 8000

# Verificación de salud del contenedor: Docker puede consultar esto
# automáticamente para saber si el servicio está realmente respondiendo,
# no solo si el proceso está vivo.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
