import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#  RNG y paths 

def crear_generador_aleatorio(semilla: int) -> np.random.Generator:
    return np.random.default_rng(semilla)

def asegurar_directorio(ruta: str | Path) -> Path:
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta

#  Helpers  tabl 

def _figsize_for_table(
    n_rows: int,
    n_cols: int,
    base_w: float = 3.0,   
    base_h: float = 0.55
):

    width  = max(12, min(28, base_w + 1.20 * n_cols))
    height = max(4.5, min(32, base_h + 0.42 * n_rows))
    return (width, height)

def guardar_dataframe_como_tabla_png(
    df,
    path_png: str | Path,
    titulo: str | None = None,
    max_rows: int | None = None,
    note: str | None = None,
    fontsize: int = 10,
    table_top: float = 0.92,
    row_scale: float = 1.32,
    title_pad: int = 16,
    wrap_headers: bool = True,
    wrap_cells: bool = False,        
    header_width_chars: int = 14,    
    cell_width_chars: int = 18        
):
    import textwrap

    path_png = Path(path_png)
    df_show = df.copy()

    truncado = False
    if max_rows is not None and len(df_show) > max_rows:
        df_show = df_show.head(max_rows).copy()
        truncado = True

    def _fmt(x):
        if isinstance(x, float):
            return f"{x:.3f}"
        return f"{x}"
    try:
        df_show = df_show.map(_fmt)
    except AttributeError:
        df_show = df_show.applymap(_fmt)

    def _wrap_text(s: str, width: int) -> str:
        return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False, replace_whitespace=False)) or str(s)

    col_labels = list(df_show.columns)
    if wrap_headers:
        col_labels_wrapped = [_wrap_text(c, header_width_chars) for c in col_labels]
    else:
        col_labels_wrapped = col_labels

    if wrap_cells:
        for c in df_show.columns:
            df_show[c] = df_show[c].apply(lambda v: _wrap_text(v, cell_width_chars))

    fig_w, fig_h = _figsize_for_table(len(df_show), len(df_show.columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=240)
    ax.axis("off")

    if titulo:
        ax.set_title(titulo, pad=title_pad, fontsize=fontsize + 4, fontweight="bold")


    tbl = ax.table(
        cellText=df_show.values,
        colLabels=col_labels_wrapped,
        loc="center",
        cellLoc="center",
        colLoc="center",
        edges="closed",
        bbox=[0.0, 0.0, 1.0, table_top]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, row_scale)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.get_text().set_wrap(True)   
        if r == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if (r % 2) else "#FBFBFB")

    import numpy as _np
    col_lengths = []
    for j, name in enumerate(col_labels_wrapped):
        header_len = len(str(name).replace("\n", " "))
        data_len = int(_np.mean([len(str(x)) for x in df_show.iloc[:min(50, len(df_show)), j].astype(str)])) if len(df_show) else 1
        col_lengths.append(max(header_len, data_len, 4))

    col_lengths = _np.array(col_lengths, dtype=float)
    widths = col_lengths / col_lengths.sum()
    widths = _np.clip(widths, 0.04, 0.20)
    widths = widths / widths.sum()

    try:
        tbl.auto_set_column_width(col=list(range(len(col_labels_wrapped))))  
    except Exception:
        pass

    for j, w in enumerate(widths):
        for i in range(len(df_show) + 1):  
            cell = tbl[i, j]
            cell.set_width(w)

    if truncado or note:
        texto = []
        if truncado: texto.append(f"Mostrando {len(df_show)} de {len(df)} filas.")
        if note:     texto.append(note)
        ax.text(0.5, 0.015, "  ·  ".join(texto), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=fontsize-1, color="#444444")

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    plt.close(fig)

#  Helpers gráf
def _pad_limits(arr, pad=0.035):
    arr = list(arr)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    span = max(1e-9, hi - lo)
    return lo - pad*span, hi + pad*span

def scatter_png(
    x, y, labels, path_png: str | Path,
    titulo: str, xlabel: str, ylabel: str,
    annotate: bool = True, fontsize: int = 11, diagonal: bool = False,
    sizes=None, dpi: int = 240,
    extra_box_text: str | None = None, box_loc: str = "upper left"
):
    path_png = Path(path_png)
    fig, ax = plt.subplots(figsize=(11, 8), dpi=dpi)
    s = sizes if sizes is not None else 40
    ax.scatter(x, y, s=s, alpha=0.9)

    if annotate:
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(str(lab), (xi, yi),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=fontsize-2, color="#1a1a1a")

    if diagonal:
        lo = min(min(x), min(y)); hi = max(max(x), max(y))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="#888888")

    xlo, xhi = _pad_limits(x, 0.035); ylo, yhi = _pad_limits(y, 0.035)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)

    ax.grid(True, linestyle=":", alpha=0.18)
    ax.set_title(titulo, fontsize=fontsize+3, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=fontsize); ax.set_ylabel(ylabel, fontsize=fontsize)

    if extra_box_text:
        locs = {"upper left":(0.02,0.98,"left","top"),"upper right":(0.98,0.98,"right","top"),
                "lower left":(0.02,0.02,"left","bottom"),"lower right":(0.98,0.02,"right","bottom")}
        x0,y0,ha,va = locs.get(box_loc, (0.02,0.98,"left","top"))
        ax.text(x0, y0, extra_box_text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize-1,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BBBBBB", alpha=0.95))

    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    plt.close(fig)

def put_figure_formula(fig: plt.Figure, text: str, loc: str = "upper left", fontsize: int = 12):
    locs = {"upper left":(0.015,0.985,"left","top"),"upper right":(0.985,0.985,"right","top"),
            "lower left":(0.015,0.015,"left","bottom"),"lower right":(0.985,0.015,"right","bottom")}
    x0,y0,ha,va = locs.get(loc, (0.015,0.985,"left","top"))
    fig.text(x0, y0, text, ha=ha, va=va, fontsize=fontsize,
             bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#BBBBBB", alpha=0.95))

def barh_diff_png(
    labels, diffs, path_png: str | Path, title: str, xlabel: str,
    se: np.ndarray | None = None, note: str | None = None, dpi: int = 240
):
    path_png = Path(path_png)
    labels = np.asarray(labels); diffs = np.asarray(diffs, dtype=float)
    order = np.argsort(diffs)
    labels_ord = labels[order]; diffs_ord = diffs[order]
    xerr = (2.0 * se[order]) if (se is not None) else None

    colors = np.where(diffs_ord >= 0, "#1f77b4", "#d62728")

    fig, ax = plt.subplots(figsize=(11, 9), dpi=dpi)
    ax.barh(range(len(labels_ord)), diffs_ord, xerr=xerr, color=colors, alpha=0.9,
            error_kw=dict(ecolor="#444444", lw=0.9, capsize=3))
    ax.axvline(0.0, color="#666666", lw=1.2, ls="--")
    ax.set_yticks(range(len(labels_ord)), labels_ord, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=12); ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.18)

    for i, v in enumerate(diffs_ord):
        ax.text(v + (0.02 if v >= 0 else -0.02)*max(1.0, abs(diffs_ord).max()),
                i, f"{v:.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color="#1a1a1a")

    if note:
        ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BBBBBB"))

    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    plt.close(fig)
