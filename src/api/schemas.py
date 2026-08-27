"""
schemas.py — Esquemas de request/response para la API de inferencia.

Usa Pydantic para validar automáticamente los datos que llegan por HTTP:
si un campo falta, tiene el tipo incorrecto, o está fuera de rango, FastAPI
devuelve un error 422 con el detalle, sin que el código del endpoint tenga
que validarlo manualmente.
"""

from pydantic import BaseModel, Field


class ClienteInput(BaseModel):
    """
    Datos de un cliente mayorista, en las mismas 6 categorías de gasto
    anual del dataset original (Wholesale Customers, UCI).

    Nota: Channel y Region NO se piden aquí a propósito — el modelo fue
    entrenado sin usarlas como input (ver README, sección "Business
    Problem"). Solo se usan para validación externa en los notebooks.
    """

    Fresh: float = Field(..., ge=0, description="Gasto anual en productos frescos")
    Milk: float = Field(..., ge=0, description="Gasto anual en lácteos")
    Grocery: float = Field(..., ge=0, description="Gasto anual en abarrotes")
    Frozen: float = Field(..., ge=0, description="Gasto anual en congelados")
    Detergents_Paper: float = Field(..., ge=0, description="Gasto anual en detergentes/papel")
    Delicassen: float = Field(..., ge=0, description="Gasto anual en delicatessen")

    class Config:
        json_schema_extra = {
            "example": {
                "Fresh": 12669,
                "Milk": 9656,
                "Grocery": 7561,
                "Frozen": 214,
                "Detergents_Paper": 2674,
                "Delicassen": 1338,
            }
        }


class PrediccionOutput(BaseModel):
    """Respuesta del endpoint /predict, siguiendo el formato de la sección M
    del enunciado para problemas de clustering."""

    cluster: int = Field(..., description="Cluster asignado (0, 1 o 2)")
    distance_to_centroid: float = Field(
        ..., description="Distancia euclidiana al centroide del cluster asignado"
    )
    model_version: str = Field(..., description="Versión del modelo en MLflow Model Registry")


class HealthOutput(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
