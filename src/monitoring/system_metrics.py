"""
system_metrics.py — Monitoreo de sistema (Sección O1): Latency, Throughput,
ErrorRate, Availability.

Guarda el estado en memoria del proceso (no en disco/DB) porque estas son
métricas operativas de corto plazo, no artefactos que necesiten trazabilidad
histórica como los runs de MLflow.

Limitación conocida: si la API corriera con múltiples workers (ej. uvicorn
--workers 4), cada worker tendría su propio estado, no compartido. Para este
proyecto (un solo proceso vía `docker run`) esto no es un problema.
"""
import time
from collections import deque

# Ventana móvil: refleja latencia reciente, no un promedio diluido desde el
# arranque del servicio (que se volvería cada vez menos representativo).
_response_times = deque(maxlen=1000)

# Contadores acumulados desde el arranque: error_rate y availability sí deben
# considerar el histórico completo, no solo la ventana reciente.
_total_requests = 0
_error_count = 0
_start_time = time.time()


def record_request(duration_seconds: float, is_error: bool = False) -> None:
    """Registra una petición completada. Llamado desde el middleware de la API."""
    global _total_requests, _error_count
    _response_times.append(duration_seconds)
    _total_requests += 1
    if is_error:
        _error_count += 1


def get_metrics() -> dict:
    """Calcula y retorna las 4 métricas de la Sección O1 en su estado actual."""
    uptime_seconds = time.time() - _start_time

    avg_latency = sum(_response_times) / len(_response_times) if _response_times else 0
    throughput = _total_requests / uptime_seconds if uptime_seconds > 0 else 0
    error_rate = (_error_count / _total_requests * 100) if _total_requests > 0 else 0
    availability = (
        (_total_requests - _error_count) / _total_requests * 100
        if _total_requests > 0 else 100  # sin requests aún = no hay evidencia de caídas
    )

    return {
        "latency_avg_ms": round(avg_latency * 1000, 2),
        "throughput_req_per_sec": round(throughput, 2),
        "error_rate_pct": round(error_rate, 2),
        "availability_pct": round(availability, 2),
        "total_requests": _total_requests,
        "uptime_seconds": round(uptime_seconds, 1),
    }