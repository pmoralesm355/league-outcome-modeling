# src/main.py
from pathlib import Path
import numpy as np
import pandas as pd

from configuracion import ConfiguracionSimulacion
from utilidades import crear_generador_aleatorio
from generador_calendario import generar_calendario_doble_vuelta
from modelos_probabilidad import simular_goles_equipo_poisson, calcular_puntos_por_resultado
from teoria import puntos_teoricos_temporada

def mu_partido(gf_home, ga_home, gf_away, ga_away, home_adv=1.05):
    mu_h = home_adv * max(0.5*(gf_home + ga_away), 1e-6)
    mu_a =          max(0.5*(gf_away + ga_home), 1e-6)
    return mu_h, mu_a

def _equipos_desde_matches(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Construye team,gf_per_match,ga_per_match desde un CSV estilo SP1:
    columnas esperadas: HomeTeam, AwayTeam, FTHG, FTAG
    """
    req = {"HomeTeam","AwayTeam","FTHG","FTAG"}
    if not req.issubset(set(map(str, df_matches.columns))):
        raise ValueError("El CSV de partidos debe contener columnas: HomeTeam, AwayTeam, FTHG, FTAG")

    home = df_matches[['HomeTeam','FTHG','FTAG']].rename(columns={'HomeTeam':'team','FTHG':'GF','FTAG':'GA'})
    away = df_matches[['AwayTeam','FTAG','FTHG']].rename(columns={'AwayTeam':'team','FTAG':'GF','FTHG':'GA'})
    full = pd.concat([home, away], ignore_index=True)
    agg = full.groupby('team').agg(matches=('GF','count'), gf=('GF','sum'), ga=('GA','sum')).reset_index()
    agg['gf_per_match'] = agg['gf'] / agg['matches']
    agg['ga_per_match'] = agg['ga'] / agg['matches']
    out = agg[['team','gf_per_match','ga_per_match']].sort_values('team').reset_index(drop=True)
    # Asegurand 20 equipos
    if len(out) != 20:
        print(f"[WARN] Se detectaron {len(out)} equipos; se continuará con todos.")
    return out

def cargar_teams(root: Path, rng: np.random.Generator) -> pd.DataFrame:
    """
    Preferencia:
      1) data/teams.csv (ya consolidado)
      2) Un CSV de partidos en data/ que contenga columnas SP1 → se construye teams
      3) Sintético (fallback)
    """
    # 1) teams.csv
    teams_csv = root / "data" / "teams.csv"
    if teams_csv.exists():
        return pd.read_csv(teams_csv)

    # 2) buscar un SP1-like en data/
    data_dir = (root / "data")
    sp1_candidate = None
    if data_dir.exists():
        for p in data_dir.glob("*.csv"):
            try:
                df_test = pd.read_csv(p, nrows=5)
                if {"HomeTeam","AwayTeam","FTHG","FTAG"}.issubset(df_test.columns):
                    sp1_candidate = p
                    break
            except Exception:
                continue
    if sp1_candidate:
        df_matches = pd.read_csv(sp1_candidate)
        teams_df = _equipos_desde_matches(df_matches)
        # guardar también como teams.csv para corridas futuras
        teams_df.to_csv(teams_csv, index=False)
        print(f"[INFO] Construido: {teams_csv}")
        return teams_df

    # 3) Fallback sintético
    print("[WARN] No se encontró data/teams.csv ni CSV de partidos compatible; se usarán equipos sintéticos.")
    n = 20
    return pd.DataFrame({
        "team": [f"Team_{i+1:02d}" for i in range(n)],
        "gf_per_match": rng.uniform(0.9, 1.9, size=n),
        "ga_per_match": rng.uniform(0.9, 1.9, size=n),
    })

def simular_temporada(teams, gf, ga, fixture, rng, home_adv=1.05):
    n = len(teams); idx = {t:i for i,t in enumerate(teams)}
    pts = np.zeros(n, dtype=int)
    for home, away in fixture:
        i, j = idx[home], idx[away]
        mu_h, mu_a = mu_partido(gf[i], ga[i], gf[j], ga[j], home_adv)
        gh, ga_ = simular_goles_equipo_poisson(mu_h, mu_a, rng)
        ph, pa = calcular_puntos_por_resultado(gh, ga_)
        pts[i] += ph; pts[j] += pa
    leaders = np.flatnonzero(pts == pts.max())
    champion = int(rng.choice(leaders))  # desempate aleatorio si corresponde
    return pts, champion

def main():
    cfg = ConfiguracionSimulacion()
    rng = crear_generador_aleatorio(cfg.semilla_aleatoria)  # UNA sola semilla → un solo RNG

    root = Path(__file__).resolve().parents[1]
    df_teams = cargar_teams(root, rng)

    teams = df_teams["team"].tolist()
    gf = df_teams["gf_per_match"].to_numpy(float)
    ga = df_teams["ga_per_match"].to_numpy(float)
    if len(teams) != 20:
        print(f"[WARN] Fixture con {len(teams)} equipos (se continúa).")

    fixture = generar_calendario_doble_vuelta(teams)

    # Teoría exacta (Skellam con scipy)
    home_adv_val = 1.05
    E_theo, V_theo = puntos_teoricos_temporada(
        teams, gf, ga, fixture, home_adv=home_adv_val
    )

    # Monte Carlo: 10,000 temporadas (un solo RNG; sin reseed)
    S = cfg.cantidad_temporadas
    n = len(teams)
    pts_all = np.zeros((n, S), dtype=int)
    champs = np.zeros(n, dtype=int)
    champ_points = np.zeros(S, dtype=int)

    for s in range(S):
        pts, champ_idx = simular_temporada(teams, gf, ga, fixture, rng, home_adv=home_adv_val)
        pts_all[:, s] = pts
        champs[champ_idx] += 1
        champ_points[s] = pts.max()

    summary = pd.DataFrame({
        "team": teams,
        "mean_points_mc": pts_all.mean(axis=1),
        "var_points_mc":  pts_all.var(axis=1, ddof=1),
        "champ_prob":     champs / S,
        "mean_points_theory": E_theo,
        "var_points_theory":  V_theo
    }).sort_values(["mean_points_mc","champ_prob"], ascending=False)

    summary["mean_diff"] = summary["mean_points_mc"] - summary["mean_points_theory"]
    summary["var_diff"]  = summary["var_points_mc"]  - summary["var_points_theory"]

    summary["seasons"] = S
    summary["avg_champion_points_mc"] = champ_points.mean()
    summary["std_champion_points_mc"] = champ_points.std(ddof=1)
    summary["seed"] = cfg.semilla_aleatoria
    summary["home_adv"] = home_adv_val
    summary["n_teams"] = len(teams)

    out_csv = root / "results_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(summary.head(10).round(3).to_string(index=False))
    print(f"\nAvg champion points (MC over {S} seasons): {champ_points.mean():.2f}")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
