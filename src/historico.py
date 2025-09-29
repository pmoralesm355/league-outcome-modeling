# src/historico.py
from __future__ import annotations
from pathlib import Path
from glob import glob
import pandas as pd

from utilidades import asegurar_directorio, guardar_dataframe_como_tabla_png

REQ = {"HomeTeam","AwayTeam","FTHG","FTAG"}

SYN = {
    "HomeTeam": ["HomeTeam","Home Team","Home","Local","EquipoLocal"],
    "AwayTeam": ["AwayTeam","Away Team","Away","Visitante","EquipoVisitante"],
    "FTHG":     ["FTHG","FT Home Goals","HomeGoals","HG","GolesLocal","FT_Home_Goals"],
    "FTAG":     ["FTAG","FT Away Goals","AwayGoals","AG","GolesVisitante","FT_Away_Goals"],
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    cols = {c.strip().lower(): c for c in df.columns}
    def find(targets):
        for t in targets:
            k = str(t).strip().lower()
            if k in cols: return cols[k]
        return None
    h = find(SYN["HomeTeam"]); a = find(SYN["AwayTeam"])
    hg = find(SYN["FTHG"]);    ag = find(SYN["FTAG"])
    if not all([h,a,hg,ag]): return None

    out = df.rename(columns={h:"HomeTeam", a:"AwayTeam", hg:"FTHG", ag:"FTAG"})
    out["FTHG"] = pd.to_numeric(out["FTHG"], errors="coerce")
    out["FTAG"] = pd.to_numeric(out["FTAG"], errors="coerce")
    out = out.dropna(subset=["HomeTeam","AwayTeam","FTHG","FTAG"]).copy()
    out["FTHG"] = out["FTHG"].astype(int)
    out["FTAG"] = out["FTAG"].astype(int)
    return out[["HomeTeam","AwayTeam","FTHG","FTAG"]]

def _leer_sp1_seguro(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "latin-1", "utf-16")
    seps = (",", ";", "\t")
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
                df.columns = [str(c).strip() for c in df.columns]
                std = _normalize_columns(df)
                if std is not None and REQ.issubset(std.columns):
                    print(f"[INFO] {path.name}: {len(std)} filas válidas (enc='{enc}', sep='{sep}')")
                    return std
            except Exception as e:
                last_err = e
                continue
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        df.columns = [str(c).strip() for c in df.columns]
        std = _normalize_columns(df)
        if std is not None and REQ.issubset(std.columns):
            print(f"[INFO] {path.name}: {len(std)} filas válidas (auto-sep)")
            return std
    except Exception as e:
        last_err = e
    raise ValueError(f"No pude estandarizar {path.name}. Último error: {last_err}")

def _agg_temporada(df_matches: pd.DataFrame) -> pd.DataFrame:
    home = df_matches[["HomeTeam","FTHG","FTAG"]].rename(columns={"HomeTeam":"team","FTHG":"GF","FTAG":"GA"})
    away = df_matches[["AwayTeam","FTAG","FTHG"]].rename(columns={"AwayTeam":"team","FTAG":"GF","FTHG":"GA"})
    full = pd.concat([home, away], ignore_index=True)
    return full.groupby("team").agg(matches=("GF","count"), gf=("GF","sum"), ga=("GA","sum")).reset_index()

def construir_historial(data_dir: Path) -> pd.DataFrame:
    archivos = sorted(glob(str(data_dir / "Season *.csv")))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos 'Season *.csv' en {data_dir}")
    acumulado, apariciones = [], {}
    for ruta in archivos:
        p = Path(ruta)
        try:
            dfm = _leer_sp1_seguro(p)
        except Exception as e:
            print(f"[WARN] Saltando {p.name}: {e}")
            continue
        agg_season = _agg_temporada(dfm)
        acumulado.append(agg_season)
        for t in set(agg_season["team"]):
            apariciones[t] = apariciones.get(t, 0) + 1
    if not acumulado:
        raise RuntimeError("No se pudo construir historial (todos los archivos fallaron).")

    big = pd.concat(acumulado, ignore_index=True)
    hist = (big.groupby("team")
               .agg(matches=("matches","sum"),
                    gf=("gf","sum"),
                    ga=("ga","sum"))
               .reset_index())
    hist["seasons_count"] = hist["team"].map(apariciones).fillna(0).astype(int)
    hist["gf_per_match"] = hist["gf"] / hist["matches"]
    hist["ga_per_match"] = hist["ga"] / hist["matches"]
    hist["gd_per_match"] = hist["gf_per_match"] - hist["ga_per_match"]
    return hist

def seleccionar_top20_consistentes(hist: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    return (hist.sort_values(["seasons_count","matches","gd_per_match"],
                             ascending=[False, False, False])
                .head(top_n)
                .loc[:, ["team","gf_per_match","ga_per_match","seasons_count","matches","gd_per_match"]]
                .reset_index(drop=True))

def cargar_top20_historico(root: Path, top_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = root / "data"
    out_dir  = asegurar_directorio(root / "output")

    hist = construir_historial(data_dir)

    guardar_dataframe_como_tabla_png(
        hist.round(3).sort_values(["seasons_count","matches","gd_per_match"], ascending=[False,False,False]),
        out_dir / "01_historical_all_teams_table.png",
        titulo="Histórico: métricas por equipo (todas las temporadas)",
        max_rows=None,
        note="Orden: seasons_count, matches, gd_per_match"
    )

    top20_full = seleccionar_top20_consistentes(hist, top_n=top_n)

    guardar_dataframe_como_tabla_png(
        top20_full.round(3),
        out_dir / "02_top20_for_simulation_table.png",
        titulo="Top-20 equipos más consistentes (usados en simulación)"
    )

    print(f"[INFO] Guardados PNGs históricos en {out_dir}")
    return hist, top20_full[["team","gf_per_match","ga_per_match"]].copy()
