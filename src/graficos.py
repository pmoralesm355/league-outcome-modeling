# src/graficos.py
"""
Genera tablas y gráficos para comparar:
- Media de puntos (MC) vs media teórica (Skellam)
- Varianza de puntos (MC) vs varianza teórica
Guarda PNGs en figs/ y muestra métricas de ajuste.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    root = Path(__file__).resolve().parents[1]
    figs = root / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    res_path = root / "results_summary.csv"
    if not res_path.exists():
        raise FileNotFoundError(f"No existe {res_path}. Ejecuta primero src/main.py")

    df = pd.read_csv(res_path)

    req_cols = {
        "team", "mean_points_mc", "var_points_mc",
        "mean_points_theory", "var_points_theory",
        "mean_diff", "var_diff", "seasons", "seed"
    }
    if not req_cols.issubset(df.columns):
        raise ValueError(f"Faltan columnas en results_summary.csv. Requeridas: {sorted(req_cols)}")

    # --- Métricas globales ---
    mae_mean = float(df["mean_diff"].abs().mean())
    mae_var  = float(df["var_diff"].abs().mean())
    print(f"[INFO] Semilla usada: {df['seed'].iloc[0]} | Temporadas: {int(df['seasons'].iloc[0])}")
    print(f"[INFO] MAE medias (|MC-Teo|): {mae_mean:.4f} | MAE varianzas: {mae_var:.4f}")

    # Orden estándar por media MC (desc) y champ_prob si existe
    sort_cols = ["mean_points_mc"]
    if "champ_prob" in df.columns:
        sort_cols.append("champ_prob")
    df = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    # --- 1) Dispersión: Media MC vs Media Teórica ---
    x = df["mean_points_theory"].to_numpy(float)
    y = df["mean_points_mc"].to_numpy(float)
    plt.figure()
    plt.scatter(x, y)
    mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
    plt.plot([mn, mx], [mn, mx])  # línea identidad
    plt.xlabel("Media teórica de puntos (Skellam)")
    plt.ylabel("Media MC de puntos (10k)")
    plt.title("Media MC vs Teórica por equipo")
    plt.tight_layout()
    out1 = figs / "mean_points_mc_vs_theory.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"[OK] Guardado: {out1}")

    # --- 2) Dispersión: Var MC vs Var Teórica ---
    x2 = df["var_points_theory"].to_numpy(float)
    y2 = df["var_points_mc"].to_numpy(float)
    plt.figure()
    plt.scatter(x2, y2)
    mn2, mx2 = min(x2.min(), y2.min()), max(x2.max(), y2.max())
    plt.plot([mn2, mx2], [mn2, mx2])  # línea identidad
    plt.xlabel("Varianza teórica de puntos")
    plt.ylabel("Varianza MC de puntos (10k)")
    plt.title("Varianza MC vs Teórica por equipo")
    plt.tight_layout()
    out2 = figs / "var_points_mc_vs_theory.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"[OK] Guardado: {out2}")

    # --- 3) Barras: diferencias de medias (MC − Teo) ---
    df_mean = df[["team","mean_diff"]].sort_values("mean_diff")
    plt.figure()
    plt.bar(df_mean["team"], df_mean["mean_diff"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Media MC − Media Teórica")
    plt.title("Diferencia de medias por equipo")
    plt.tight_layout()
    out3 = figs / "mean_diff_bars.png"
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"[OK] Guardado: {out3}")

    # --- 4) Barras: diferencias de varianzas (MC − Teo) ---
    df_var = df[["team","var_diff"]].sort_values("var_diff")
    plt.figure()
    plt.bar(df_var["team"], df_var["var_diff"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Var MC − Var Teórica")
    plt.title("Diferencia de varianzas por equipo")
    plt.tight_layout()
    out4 = figs / "var_diff_bars.png"
    plt.savefig(out4, dpi=150)
    plt.close()
    print(f"[OK] Guardado: {out4}")

    # --- 5) Export de tabla comparativa ordenada ---
    out_csv = root / "comparison_mc_vs_theory.csv"
    df_out = df[[
        "team",
        "mean_points_theory","mean_points_mc","mean_diff",
        "var_points_theory","var_points_mc","var_diff"
    ]]
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] Guardado CSV comparativo: {out_csv}")

if __name__ == "__main__":
    main()
