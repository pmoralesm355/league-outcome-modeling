# src/main.py
from pathlib import Path
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from configuracion import ConfiguracionSimulacion
from utilidades import (
    crear_generador_aleatorio, asegurar_directorio,
    guardar_dataframe_como_tabla_png, scatter_png, put_figure_formula,
    barh_diff_png
)
from generador_calendario import generar_calendario_doble_vuelta
from modelos_probabilidad import simular_goles_equipo_poisson, calcular_puntos_por_resultado
from teoria import puntos_teoricos_temporada
from historico import cargar_top20_historico  # multi-temporada

def mu_partido(gf_home, ga_home, gf_away, ga_away, home_adv=1.05):
    mu_h = home_adv * max(0.5 * (gf_home + ga_away), 1e-6)
    mu_a =           max(0.5 * (gf_away + ga_home), 1e-6)
    return mu_h, mu_a

def simular_temporada(teams, gf, ga, fixture, rng, home_adv=1.05):
    n = len(teams); idx = {t: i for i, t in enumerate(teams)}
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

def _seed_non_repeating() -> int:
    # semilla no repetible por ejecución
    return int.from_bytes(os.urandom(8), "little")

def main():
    cfg = ConfiguracionSimulacion()
    root = Path(__file__).resolve().parents[1]
    out_dir = asegurar_directorio(root / "output")

    # semilla distinta por ejecución
    seed = _seed_non_repeating()
    rng = crear_generador_aleatorio(seed)
    print(f"[INFO] Seed used: {seed}")

    # 1) Histórico y Top-20
    print(f"[INFO] Buscando 'Season *.csv' en {root/'data'}...")
    _, df_top20 = cargar_top20_historico(root, top_n=20)

    teams = df_top20["team"].tolist()
    gf = df_top20["gf_per_match"].to_numpy(float)
    ga = df_top20["ga_per_match"].to_numpy(float)
    if len(teams) != 20:
        print(f"[WARN] Fixture con {len(teams)} equipos (se continúa).")

    fixture = generar_calendario_doble_vuelta(teams)

    # 2) Teoría exacta (Skellam)
    home_adv_val = 1.05
    E_theo, V_theo = puntos_teoricos_temporada(teams, gf, ga, fixture, home_adv=home_adv_val)

    # 3) Monte Carlo: 10,000 temporadas
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

    # 4) Resumen MC vs Teoría + meta
    summary = pd.DataFrame({
        "team": teams,
        "mean_points_mc": pts_all.mean(axis=1),
        "var_points_mc":  pts_all.var(axis=1, ddof=1),
        "champ_prob":     champs / S,
        "mean_points_theory": E_theo,
        "var_points_theory":  V_theo
    }).sort_values(["mean_points_mc", "champ_prob"], ascending=False)

    summary["mean_diff"] = summary["mean_points_mc"] - summary["mean_points_theory"]
    summary["var_diff"]  = summary["var_points_mc"]  - summary["var_points_theory"]
    summary["seasons"] = S
    summary["avg_champion_points_mc"] = champ_points.mean()
    summary["std_champion_points_mc"] = champ_points.std(ddof=1)
    summary["seed"] = seed
    summary["home_adv"] = home_adv_val
    summary["n_teams"] = len(teams)

    #  ARCHIVOS 
    summary_csv = out_dir / "simulation_results_summary.csv"
    summary.round(6).to_csv(summary_csv, index=False)

    # (3) Tabla resumen
    cols_order = [
        "team", "mean_points_mc", "var_points_mc", "champ_prob",
        "mean_points_theory", "var_points_theory",
        "mean_diff", "var_diff",
        "seasons", "avg_champion_points_mc", "std_champion_points_mc",
        "seed", "home_adv", "n_teams"
    ]
    pretty_cols = {
        "team": "Equipo",
        "mean_points_mc": "Media MC",
        "var_points_mc": "Var MC",
        "champ_prob": "P(campeón)",
        "mean_points_theory": "Media Teoría",
        "var_points_theory": "Var Teoría",
        "mean_diff": "Δ Media (MC−T)",
        "var_diff": "Δ Var (MC−T)",
        "seasons": "Temporadas",
        "avg_champion_points_mc": "Media pts campeón (MC)",
        "std_champion_points_mc": "Desv. pts campeón (MC)",
        "seed": "Semilla",
        "home_adv": "Ventaja local",
        "n_teams": "# equipos",
    }

    summary_display = summary[cols_order].rename(columns=pretty_cols)

    note_text = (
        f"Temporadas: {S:,}  ·  Avg pts campeón (MC): {champ_points.mean():.2f}  ·  "
        f"Desv: {champ_points.std(ddof=1):.3f}  ·  Seed: {seed}  ·  "
        f"Ventaja local: {home_adv_val:.3f}  ·  #equipos: {len(teams)}"
    )

    guardar_dataframe_como_tabla_png(
        summary_display.round({
            "Media MC": 3, "Var MC": 3, "P(campeón)": 3,
            "Media Teoría": 3, "Var Teoría": 3,
            "Δ Media (MC−T)": 3, "Δ Var (MC−T)": 3,
            "Media pts campeón (MC)": 3, "Desv. pts campeón (MC)": 3,
            "Ventaja local": 2
        }),
        out_dir / "03_simulation_summary_table.png",
        titulo="Simulación (10,000 temporadas): MC vs Teoría",
        max_rows=None,
        fontsize=10,          
        table_top=0.84,       
        row_scale=1.28,       
        title_pad=24          
    )

    # (4) MC: media vs varianza (dispersión)
    mc_box = r"$S=10{,}000$ temporadas" "\n" \
             r"$\mu_i^{\rm MC}=\frac{1}{S}\sum_s P_i^{(s)}$" "\n" \
             r"$\sigma_{i,\rm MC}^2=\mathrm{Var}_s(P_i^{(s)})$"
    scatter_png(
        x=summary["mean_points_mc"].tolist(),
        y=summary["var_points_mc"].tolist(),
        labels=summary["team"].tolist(),
        path_png=out_dir / "04_mc_mean_vs_variance_scatter.png",
        titulo="Monte Carlo: Media vs Varianza de puntos por equipo",
        xlabel="Media de puntos (MC)", ylabel="Varianza de puntos (MC)",
        annotate=True, diagonal=False, extra_box_text=mc_box, box_loc="upper left"
    )

    # (5) Teoría: media vs varianza (dispersión con fórmulas fuera del eje)
    fig5, ax5 = plt.subplots(figsize=(11, 8), dpi=240)
    ax5.scatter(summary["mean_points_theory"], summary["var_points_theory"])
    for xt, yt, lab in zip(summary["mean_points_theory"], summary["var_points_theory"], summary["team"]):
        ax5.annotate(str(lab), (xt, yt), xytext=(3, 3), textcoords="offset points", fontsize=9)
    ax5.set_title("Teoría: Media vs Varianza de puntos por equipo", fontsize=14, fontweight="bold")
    ax5.set_xlabel("Media de puntos (Teoría)")
    ax5.set_ylabel("Varianza de puntos (Teoría)")

    teoria_box = (
        r"$D=G_h-G_a\sim{\rm Skellam}(\mu_h,\mu_a)$" "\n"
        r"$p_W=1-F(0),\; p_D=f(0),\; p_L=F(-1)$" "\n"
        r"$E=3p_W+1p_D,\quad {\rm Var}=\sum_{x\in\{0,1,3\}} (x-E)^2P(X=x)$" "\n"
        r"${\rm Temporada:}\ E_i=\sum_m E_{i,m},\ {\rm Var}_i=\sum_m {\rm Var}_{i,m}$"
    )
    put_figure_formula(fig5, teoria_box, loc="upper left", fontsize=12)
    fig5.tight_layout()
    fig5.savefig(out_dir / "05_theory_mean_vs_variance_scatter.png")
    plt.close(fig5)

    # (6) Comparación MC vs Teoría (dos paneles)
    fig6, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=240)

    # Medias
    axL = axes[0]
    axL.scatter(summary["mean_points_theory"], summary["mean_points_mc"])
    for xt, xm, lab in zip(summary["mean_points_theory"], summary["mean_points_mc"], summary["team"]):
        axL.annotate(str(lab), (xt, xm), xytext=(3, 3), textcoords="offset points", fontsize=9)
    lo = float(min(summary["mean_points_theory"].min(), summary["mean_points_mc"].min()))
    hi = float(max(summary["mean_points_theory"].max(), summary["mean_points_mc"].max()))
    axL.plot([lo, hi], [lo, hi], "--", linewidth=1.2, color="#888888")
    axL.grid(True, linestyle=":", alpha=0.18)
    axL.set_title("Media: MC vs Teoría", fontweight="bold")
    axL.set_xlabel("Media (Teoría)"); axL.set_ylabel("Media (MC)")

    mu_mc = summary["mean_points_mc"].to_numpy()
    mu_th = summary["mean_points_theory"].to_numpy()
    r_mu = np.corrcoef(mu_th, mu_mc)[0,1]
    mae_mu = float(np.mean(np.abs(mu_mc - mu_th)))
    rmse_mu = float(np.sqrt(np.mean((mu_mc - mu_th)**2)))
    bias_mu = float(np.mean(mu_mc - mu_th))
    box_mu = f"r={r_mu:.3f}\nMAE={mae_mu:.3f}\nRMSE={rmse_mu:.3f}\nSesgo(μ_MC-μ_Teor)={bias_mu:.3f}"
    axL.text(0.02, 0.02, box_mu, transform=axL.transAxes, ha="left", va="bottom",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", ec="#BBBBBB"))

    # Varianzas
    axR = axes[1]
    axR.scatter(summary["var_points_theory"], summary["var_points_mc"])
    for vt, vm, lab in zip(summary["var_points_theory"], summary["var_points_mc"], summary["team"]):
        axR.annotate(str(lab), (vt, vm), xytext=(3, 3), textcoords="offset points", fontsize=9)
    lo_v = float(min(summary["var_points_theory"].min(), summary["var_points_mc"].min()))
    hi_v = float(max(summary["var_points_theory"].max(), summary["var_points_mc"].max()))
    axR.plot([lo_v, hi_v], [lo_v, hi_v], "--", linewidth=1.2, color="#888888")
    axR.grid(True, linestyle=":", alpha=0.18)
    axR.set_title("Varianza: MC vs Teoría", fontweight="bold")
    axR.set_xlabel("Varianza (Teoría)"); axR.set_ylabel("Varianza (MC)")

    v_mc = summary["var_points_mc"].to_numpy()
    v_th = summary["var_points_theory"].to_numpy()
    r_v = np.corrcoef(v_th, v_mc)[0,1]
    mae_v = float(np.mean(np.abs(v_mc - v_th)))
    rmse_v = float(np.sqrt(np.mean((v_mc - v_th)**2)))
    bias_v = float(np.mean(v_mc - v_th))
    box_v = f"r={r_v:.3f}\nMAE={mae_v:.3f}\nRMSE={rmse_v:.3f}\nSesgo(σ²_MC-σ²_Teor)={bias_v:.3f}"
    axR.text(0.98, 0.02, box_v, transform=axR.transAxes, ha="right", va="bottom",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", ec="#BBBBBB"))

    fig6.suptitle("Comparación MC vs Teoría", fontsize=14, fontweight="bold")
    fig6.tight_layout()
    fig6.savefig(out_dir / "06_comparison_mc_vs_theory.png")
    plt.close(fig6)

    # (7) Barras — diferencias de medias (MC - Teoría) con ±2·SE
    se_mu = np.sqrt(summary["var_points_mc"].to_numpy() / S)
    barh_diff_png(
        labels=summary["team"].tolist(),
        diffs=summary["mean_diff"].to_numpy(),
        se=se_mu,
        path_png=out_dir / "07_mean_diff_bars.png",
        title="Diferencia de medias por equipo (MC − Teoría)",
        xlabel="Puntos de liga (diferencia)",
        note="Error bars: ±2·SE,  SE = √(Var_MC / S)"
    )

    # (8) Barras — diferencias de varianzas (MC - Teoría) con ±2·SE (aprox)
    se_var = summary["var_points_mc"].to_numpy() * np.sqrt(2.0 / max(1, S - 1))
    barh_diff_png(
        labels=summary["team"].tolist(),
        diffs=summary["var_diff"].to_numpy(),
        se=se_var,
        path_png=out_dir / "08_var_diff_bars.png",
        title="Diferencia de varianzas por equipo (MC − Teoría)",
        xlabel="Varianza de puntos (diferencia)",
        note="Error bars: ±2·SE,  SE ≈ σ²·√(2/(S−1))"
    )

    print("[OK] PNGs generados en", out_dir)
    for f in [
        "01_historical_all_teams_table.png",
        "02_top20_for_simulation_table.png",
        "03_simulation_summary_table.png",
        "04_mc_mean_vs_variance_scatter.png",
        "05_theory_mean_vs_variance_scatter.png",
        "06_comparison_mc_vs_theory.png",
        "07_mean_diff_bars.png",
        "08_var_diff_bars.png",
    ]:
        print("  -", f)
    print("CSV:", summary_csv)

if __name__ == "__main__":
    main()
