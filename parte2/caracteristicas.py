import numpy as np
import pandas as pd
from collections import Counter

def construir_caracteristicas_mano(fila: pd.Series) -> pd.Series:
    suits = [fila[f"S{i}"] for i in range(1,6)]
    ranks = [fila[f"R{i}"] for i in range(1,6)]
    ranks_sorted = sorted(ranks)
    uniq_ranks = len(set(ranks))
    uniq_suits = len(set(suits))

    c = Counter(ranks)
    counts = sorted(c.values(), reverse=True)
    max_dup = counts[0]
    n_pares = sum(1 for v in counts if v == 2)
    hay_trio = 1 if 3 in counts else 0
    hay_poker = 1 if 4 in counts else 0

    es_flush = 1 if uniq_suits == 1 else 0
    es_straight = 1 if (uniq_ranks == 5 and (max(ranks_sorted)-min(ranks_sorted) == 4)) else 0

    rango_span = max(ranks_sorted) - min(ranks_sorted)
    gaps = np.diff(ranks_sorted)
    suma_gaps = float(np.sum(gaps))
    gap_max = float(np.max(gaps))

    hist_rangos = [0]*13
    for r in ranks:
        hist_rangos[r-1] += 1

    r1,r2,r3,r4,r5 = ranks_sorted

    base = {
        "uniq_rangos": uniq_ranks,
        "uniq_palos": uniq_suits,
        "max_duplicados": max_dup,
        "num_pares": n_pares,
        "hay_trio": hay_trio,
        "hay_poker": hay_poker,
        "es_flush": es_flush,
        "es_straight": es_straight,
        "rango_span": rango_span,
        "suma_gaps": suma_gaps,
        "gap_max": gap_max,
        "r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5,
    }
    hist = {f"hist_rango_{i+1}": hist_rangos[i] for i in range(13)}
    return pd.Series({**base, **hist})

def construir_matriz_caracteristicas(df: pd.DataFrame):
    X = df.apply(construir_caracteristicas_mano, axis=1)
    y = df["label"].astype(int).values
    return X, y
