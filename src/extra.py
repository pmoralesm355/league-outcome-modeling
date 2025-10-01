# src/extra.py
# Simular LaLiga con SOLO la última temporada y comparar con la tabla real.
# Usa "Last Season 2024-2025.csv"
# Home advantage calibrada de esa temporada: mean(FTHG)/mean(FTAG) ∈ [0.80, 1.30].

from __future__ import annotations
from pathlib import Path
from glob import glob
import os
import numpy as np
import pandas as pd

from configuracion import ConfiguracionSimulacion
from utilidades import (
    crear_generador_aleatorio, asegurar_directorio,
    guardar_dataframe_como_tabla_png
)
from generador_calendario import generar_calendario_doble_vuelta
from modelos_probabilidad import simular_goles_equipo_poisson, calcular_puntos_por_resultado

#  Lectura robusta de “Last Season” 
REQ = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
SYN = {
    "HomeTeam": ["HomeTeam","Home Team","Home","Local","EquipoLocal"],
    "AwayTeam": ["AwayTeam","Away Team","Away","Visitante","EquipoVisitante"],
    "FTHG":     ["FTHG","FT Home Goals","HomeGoals","HG","GolesLocal","FT_Home_Goals"],
    "FTAG":     ["FTAG","FT Away Goals","AwayGoals","AG","GolesVisitante","FT_Away_Goals"],
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    cols = {str(c).strip().lower(): c for c in df.columns}
    def _find(targets):
        for t in targets:
            k = str(t).strip().lower()
            if k in cols: return cols[k]
        return None
    h = _find(SYN["HomeTeam"]); a = _find(SYN["AwayTeam"])
    hg = _find(SYN["FTHG"]);    ag = _find(SYN["FTAG"])
    if not all([h, a, hg, ag]): return None
    out = df.rename(columns={h:"HomeTeam", a:"AwayTeam", hg:"FTHG", ag:"FTAG"}).copy()
    out["FTHG"] = pd.to_numeric(out["FTHG"], errors="coerce")
    out["FTAG"] = pd.to_numeric(out["FTAG"], errors="coerce")
    out = out.dropna(subset=["HomeTeam","AwayTeam","FTHG","FTAG"])
    out["FTHG"] = out["FTHG"].astype(int); out["FTAG"] = out["FTAG"].astype(int)
    return out[["HomeTeam","AwayTeam","FTHG","FTAG"]]

def leer_last_season_matches(path_csv: Path) -> pd.DataFrame:
    encodings = ("utf-8","latin-1","utf-16"); seps = (",",";","\t"); last_err=None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path_csv, encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
                df.columns = [str(c).strip() for c in df.columns]
                std = _normalize_columns(df)
                if std is not None and REQ.issubset(std.columns):
                    print(f"[INFO] {path_csv.name}: {len(std)} filas válidas (enc='{enc}', sep='{sep}')")
                    return std
            except Exception as e:
                last_err = e; continue
    try:
        df = pd.read_csv(path_csv, engine="python", on_bad_lines="skip")
        df.columns = [str(c).strip() for c in df.columns]
        std = _normalize_columns(df)
        if std is not None and REQ.issubset(std.columns):
            print(f"[INFO] {path_csv.name}: {len(std)} filas válidas (auto-sep)")
            return std
    except Exception as e:
        last_err = e
    raise ValueError(f"No pude estandarizar {path_csv.name}. Último error: {last_err}")

#  Transformaciones de temporada 
def tabla_real_y_tasas(df_matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    home_pts = np.where(df_matches["FTHG"].values > df_matches["FTAG"].values, 3,
                 np.where(df_matches["FTHG"].values == df_matches["FTAG"].values, 1, 0))
    away_pts = np.where(df_matches["FTAG"].values > df_matches["FTHG"].values, 3,
                 np.where(df_matches["FTAG"].values == df_matches["FTHG"].values, 1, 0))
    home = pd.DataFrame({"team": df_matches["HomeTeam"].values,
                         "GF": df_matches["FTHG"].values,
                         "GA": df_matches["FTAG"].values,
                         "Pts": home_pts})
    away = pd.DataFrame({"team": df_matches["AwayTeam"].values,
                         "GF": df_matches["FTAG"].values,
                         "GA": df_matches["FTHG"].values,
                         "Pts": away_pts})
    full = pd.concat([home, away], ignore_index=True)
    tabla = (full.groupby("team")
                   .agg(matches=("GF","count"), gf=("GF","sum"), ga=("GA","sum"), points=("Pts","sum"))
                   .reset_index())
    tabla["gd"] = tabla["gf"] - tabla["ga"]
    tabla = tabla.sort_values(["points","gd","gf"], ascending=[False,False,False]).reset_index(drop=True)
    tasas = tabla[["team","gf","ga","matches"]].copy()
    tasas["gf_per_match"] = tasas["gf"] / tasas["matches"]
    tasas["ga_per_match"] = tasas["ga"] / tasas["matches"]
    return tabla, tasas[["team","gf_per_match","ga_per_match"]]

def calibrar_home_adv(df_matches: pd.DataFrame) -> float:
    h_mean = float(df_matches["FTHG"].mean()); a_mean = float(df_matches["FTAG"].mean())
    if a_mean <= 0: return 1.0
    return float(max(0.80, min(1.30, h_mean / a_mean)))

#  Motor de simulación (Poisson para goles) 
def mu_partido(gf_home, ga_home, gf_away, ga_away, home_adv=1.05):
    mu_h = home_adv * max(0.5 * (gf_home + ga_away), 1e-6)
    mu_a =           max(0.5 * (gf_away + ga_home), 1e-6)
    return mu_h, mu_a

def simular_temporada(teams, gf, ga, fixture, rng, home_adv=1.05):
    idx = {t: i for i, t in enumerate(teams)}
    pts = np.zeros(len(teams), dtype=int)
    for home, away in fixture:
        i, j = idx[home], idx[away]
        mu_h, mu_a = mu_partido(gf[i], ga[i], gf[j], ga[j], home_adv)
        gh, ga_ = simular_goles_equipo_poisson(mu_h, mu_a, rng)  # Poisson independientes
        ph, pa = calcular_puntos_por_resultado(gh, ga_)
        pts[i] += ph; pts[j] += pa
    leaders = np.flatnonzero(pts == pts.max())
    return pts, int(rng.choice(leaders))

def _seed_non_repeating() -> int:
    return int.from_bytes(os.urandom(8), "little")

#  Pipeline principal 
def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    out_dir  = asegurar_directorio(root / "output")

    prefer = data_dir / "Last Season 2024-2025.csv"
    if prefer.exists():
        last_csv = prefer
    else:
        cand = sorted(glob(str(data_dir / "Last Season *.csv")))
        if not cand:
            raise FileNotFoundError(f"No encontré 'Last Season *.csv' en {data_dir}")
        last_csv = Path(cand[0])
    print(f"[INFO] Usando archivo: {last_csv.name}")

    dfm = leer_last_season_matches(last_csv)
    tabla_real, tasas = tabla_real_y_tasas(dfm)
    home_adv_val = calibrar_home_adv(dfm)

    teams = tabla_real["team"].tolist()
    t_idx = tasas.set_index("team")
    gf = t_idx.loc[teams, "gf_per_match"].to_numpy(float)
    ga = t_idx.loc[teams, "ga_per_match"].to_numpy(float)
    fixture = generar_calendario_doble_vuelta(teams)

    S = ConfiguracionSimulacion().cantidad_temporadas
    seed = _seed_non_repeating()
    rng = crear_generador_aleatorio(seed)
    print(f"[INFO] Seed used: {seed}")

    n = len(teams)
    pts_all = np.zeros((n, S), dtype=int)
    champs = np.zeros(n, dtype=int)
    for s in range(S):
        pts, k = simular_temporada(teams, gf, ga, fixture, rng, home_adv=home_adv_val)
        pts_all[:, s] = pts
        champs[k] += 1

    mc_mean = pts_all.mean(axis=1)
    mc_var  = pts_all.var(axis=1, ddof=1)
    champ_prob = champs / S

    orden_real = tabla_real.sort_values(["points","gd","gf"], ascending=[False,False,False])["team"].tolist()
    rank_real_map = {t: i+1 for i, t in enumerate(orden_real)}
    rank_real = [rank_real_map[t] for t in teams]

    order_idx = np.lexsort((-champ_prob, -mc_mean))
    rank_mc = np.empty(n, dtype=int); rank_mc[order_idx[::-1]] = np.arange(1, n+1)

    comp = pd.DataFrame({
        "team": teams,
        "points_real": tabla_real.set_index("team").loc[teams, "points"].to_numpy(int),
        "mean_points_mc": mc_mean,
        "var_points_mc": mc_var,
        "champ_prob": champ_prob,
        "rank_real": rank_real,
        "rank_mc": rank_mc,
    })
    comp["diff_mean"] = comp["mean_points_mc"] - comp["points_real"]
    comp["rank_diff"] = comp["rank_mc"] - comp["rank_real"]

    r = float(np.corrcoef(comp["points_real"], comp["mean_points_mc"])[0,1])
    mae = float(np.mean(np.abs(comp["mean_points_mc"] - comp["points_real"])))
    rmse = float(np.sqrt(np.mean((comp["mean_points_mc"] - comp["points_real"])**2)))
    bias = float(np.mean(comp["mean_points_mc"] - comp["points_real"]))

    out_dir = asegurar_directorio(out_dir)
    csv_path = out_dir / "extra_last_season_comparison.csv"
    comp_csv = comp.copy()
    comp_csv[["mean_points_mc","var_points_mc","champ_prob","diff_mean"]] = \
        comp_csv[["mean_points_mc","var_points_mc","champ_prob","diff_mean"]].round(3)
    comp_csv.to_csv(csv_path, index=False)

    note = (f"LaLiga 24/25 | S={S} | home_adv={home_adv_val:.3f} | seed={seed} | "
            f"r={r:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f} | Sesgo={bias:.3f}")

    cols = ["team","points_real","mean_points_mc","var_points_mc","champ_prob",
            "diff_mean","rank_real","rank_mc","rank_diff"]
    comp_png = (comp_csv[cols]
                .sort_values(["points_real","mean_points_mc"], ascending=[False, False])
                .rename(columns={
                    "points_real":"Pts_real",
                    "mean_points_mc":"μ_MC",
                    "var_points_mc":"Var_MC",
                    "champ_prob":"P(champ)",
                    "diff_mean":"Δμ",
                    "rank_real":"R_real",
                    "rank_mc":"R_MC",
                    "rank_diff":"ΔR"
                }))

    guardar_dataframe_como_tabla_png(
        comp_png,
        out_dir / "extra_last_season_comparison.png",
        titulo="LaLiga 24/25: 10k simulaciones vs tabla real",
        max_rows=None,
        note=note,
        fontsize=9
    )

    print("[OK] Generados archivos en", out_dir)
    print(" - extra_last_season_comparison.csv")
    print(" - extra_last_season_comparison.png")

if __name__ == "__main__":
    main()
