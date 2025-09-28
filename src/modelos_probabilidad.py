# src/modelos_probabilidad.py
from typing import Optional, Tuple
import numpy as np

def simular_goles_equipo_poisson(
    promedio_local: float, promedio_visitante: float, generador: np.random.Generator
) -> Tuple[int, int]:
    """Genera goles (local, visita) con Poisson independientes."""
    goles_local = int(generador.poisson(promedio_local))
    goles_visitante = int(generador.poisson(promedio_visitante))
    return goles_local, goles_visitante

def simular_tiempo_primer_gol_exponencial(
    promedio_local: float, promedio_visitante: float,
    minutos_partido: int, generador: np.random.Generator
) -> Tuple[Optional[float], Optional[bool]]:
    """
    Tiempo al primer gol ~ Exponencial(λ_t), λ_t=(μ_local+μ_visita)/minutos.
    Devuelve (tiempo, es_local). Si censura (sin gol), devuelve (None, None).
    """
    tasa_total = promedio_local + promedio_visitante
    if tasa_total <= 0:
        return None, None
    tasa_por_minuto = tasa_total / minutos_partido
    tiempo = float(generador.exponential(1.0 / tasa_por_minuto))
    if tiempo > minutos_partido:
        return None, None  # censura
    prob_primer_gol_local = promedio_local / tasa_total
    es_local = bool(generador.random() < prob_primer_gol_local)
    return tiempo, es_local

def calcular_puntos_por_resultado(goles_local: int, goles_visitante: int) -> Tuple[int, int]:
    """Asigna puntos (3/1/0) según el resultado."""
    if goles_local > goles_visitante: return 3, 0
    if goles_local < goles_visitante: return 0, 3
    return 1, 1

#  MLE Exponencial con censura (para ddeterminar la distro del tiempo al primer gol)
def mle_lambda_tiempo_censurado(t_obs, n_sin_gol: int, corte: float = 90.0) -> float:
    """
    λ̂_t = d / (sum(t_i) + n_sin_gol*corte)
    d = #con gol observado. Cada sin gol aporta 'corte' minutos (censura derecha).
    """
    t = np.asarray(t_obs, float)
    d = int(t.size)
    total = float(t.sum()) + float(n_sin_gol) * float(corte)
    return (d / total) if total > 0 else 0.0
