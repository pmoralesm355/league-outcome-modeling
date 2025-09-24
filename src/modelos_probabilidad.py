from typing import Optional, Tuple
import numpy as np

def simular_goles_equipo_poisson(promedio_local: float, promedio_visitante: float,
                                 generador: np.random.Generator) -> Tuple[int, int]:
    """
    Genera el número de goles de local y visitante en un partido,
    """
    goles_local = int(generador.poisson(promedio_local))
    goles_visitante = int(generador.poisson(promedio_visitante))
    return goles_local, goles_visitante

def simular_tiempo_primer_gol_exponencial(promedio_local: float, promedio_visitante: float,
                                          minutos_partido: int,
                                          generador: np.random.Generator
                                          ) -> Tuple[Optional[float], Optional[bool]]:
    """
    Devuelve el tiempo hasta el primer gol y si el gol fue del equipo local,
    """
    tasa_total = promedio_local + promedio_visitante
    if tasa_total <= 0:
        return None, None
    tasa_por_minuto = tasa_total / minutos_partido
    tiempo = float(generador.exponential(1.0 / tasa_por_minuto))
    if tiempo > minutos_partido:
        return None, None
    prob_primer_gol_local = promedio_local / tasa_total
    es_local = bool(generador.random() < prob_primer_gol_local)
    return tiempo, es_local

def calcular_puntos_por_resultado(goles_local: int, goles_visitante: int) -> Tuple[int, int]:
    """Asigna puntos (3,1,0) según el resultado del partido."""
    if goles_local > goles_visitante:
        return 3, 0
    if goles_local < goles_visitante:
        return 0, 3
    return 1, 1
