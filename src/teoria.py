# src/teoria.py
from __future__ import annotations
import numpy as np
from scipy.stats import skellam  #

def probs_resultado(mu_h: float, mu_a: float) -> tuple[float, float, float]:
    """
    Probabilidades exactas (Skellam) de resultado del LOCAL:
      pW = P(D>0), pD = P(D=0), pL = P(D<0) con D = G_h - G_a ~ Skellam(mu_h, mu_a).
    """
    pD = float(skellam.pmf(0, mu_h, mu_a))
    pW = float(1.0 - skellam.cdf(0, mu_h, mu_a))
    pL = float(skellam.cdf(-1, mu_h, mu_a))
    return pW, pD, pL

def puntos_teoricos_temporada(
    teams: list[str], gf: np.ndarray, ga: np.ndarray,
    fixture: list[tuple[str, str]], home_adv: float = 1.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    E y Var teóricas de puntos por equipo sumando todos los partidos del fixture.
    """
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    E = np.zeros(n); V = np.zeros(n)
    for home, away in fixture:
        i, j = idx[home], idx[away]
        # Misma regla de intensidades que en la simulación
        mu_h = home_adv * max(0.5*(gf[i] + ga[j]), 1e-6)
        mu_a =          max(0.5*(gf[j] + ga[i]), 1e-6)

        pW, pD, pL = probs_resultado(mu_h, mu_a)

        # Local
        E_h = 3*pW + 1*pD
        V_h = (0-E_h)**2 * pL + (1-E_h)**2 * pD + (3-E_h)**2 * pW
        E[i] += E_h; V[i] += V_h
        # Visita: invierte win/loss
        E_a = 3*pL + 1*pD
        V_a = (0-E_a)**2 * pW + (1-E_a)**2 * pD + (3-E_a)**2 * pL
        E[j] += E_a; V[j] += V_a
    return E, V
