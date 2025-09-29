import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from configuracion import TODAS_LAS_CLASES, NOMBRES_CLASES

def graficar_distribucion_clases(y, titulo="Distribución de clases"):
    cuentas = pd.Series(y).value_counts().reindex(TODAS_LAS_CLASES, fill_value=0)
    total = cuentas.sum()
    plt.figure(figsize=(10,4))
    barras = plt.bar(range(len(TODAS_LAS_CLASES)), cuentas.values)
    plt.xticks(range(len(TODAS_LAS_CLASES)), [NOMBRES_CLASES[i] for i in TODAS_LAS_CLASES],
               rotation=35, ha="right")
    plt.ylabel("Frecuencia (n)")
    plt.title(titulo)
    for b, c in zip(barras, cuentas.values):
        pct = 100.0*c/total if total else 0.0
        plt.text(b.get_x()+b.get_width()/2, b.get_height(), f"{c}\n({pct:.2f}%)",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.show()

def graficar_confusion_apilada(cm, titulo, top_k=5):
    cm = cm.astype(float)
    suma_filas = cm.sum(axis=1, keepdims=True); suma_filas[suma_filas == 0] = 1.0
    cmn = cm / suma_filas

    datos = []
    for i, fila in enumerate(cmn):
        orden = np.argsort(fila)[::-1]
        top = orden[:top_k]
        otros = fila[orden[top_k:]].sum() if len(orden) > top_k else 0.0
        for j in top:  datos.append({"real": i, "pred": j, "pred_name": f"{j}", "pct": fila[j]})
        if otros > 0: datos.append({"real": i, "pred": -1, "pred_name": "otros", "pct": otros})

    df = pd.DataFrame(datos)
    df["real_name"] = df["real"].map(lambda k: NOMBRES_CLASES[k] + f" (n={int(cm[k].sum())})")
    pv = df.pivot_table(index="real_name", columns="pred_name", values="pct", aggfunc="sum").fillna(0.0)
    orden_filas = [NOMBRES_CLASES[i] + f" (n={int(cm[i].sum())})" for i in TODAS_LAS_CLASES]
    pv = pv.reindex(orden_filas)

    cols = [str(i) for i in TODAS_LAS_CLASES if str(i) in pv.columns]
    if "otros" in pv.columns: cols.append("otros")
    cols = [c for c in cols if c in pv.columns] + [c for c in pv.columns if c not in cols]
    pv = pv[cols]

    ax = pv.plot(kind="barh", stacked=True, figsize=(10,7), width=0.85)
    ax.set_xlabel("Porcentaje dentro de la clase real")
    ax.set_ylabel("Clase real (con soporte)")
    ax.set_title(titulo)
    ax.set_xlim(0, 1)
    for cont in ax.containers:
        for bar in cont:
            w = bar.get_width()
            if w > 0.04:
                ax.text(bar.get_x()+w/2, bar.get_y()+bar.get_height()/2, f"{w*100:.1f}%",
                        ha="center", va="center", fontsize=8)
    ax.legend(title="Predicho", ncol=3, bbox_to_anchor=(1.0, 1.02), loc="lower right")
    plt.tight_layout(); plt.show()

def graficar_metricas_por_clase(y_true, y_pred, titulo):
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=TODAS_LAS_CLASES, zero_division=0
    )
    x = np.arange(len(TODAS_LAS_CLASES)); width = 0.28
    plt.figure(figsize=(11,5.5))
    plt.bar(x - width, prec, width, label="Precision")
    plt.bar(x,         rec,  width, label="Recall")
    plt.bar(x + width, f1,   width, label="F1")
    plt.xticks(x, [f"{i}" for i in TODAS_LAS_CLASES]); plt.ylim(0, 1.05)
    plt.xlabel("Clase"); plt.ylabel("Score"); plt.title(titulo)
    for i, s in enumerate(sup.astype(int)):
        plt.text(i, 1.03, f"n={s}", ha="center", va="bottom", fontsize=8)
    plt.legend(loc="lower right"); plt.tight_layout(); plt.show()

def graficar_resumen_modelos(macro_f1, bal_acc, nombres_modelos):
    x = np.arange(len(nombres_modelos)); width = 0.35
    plt.figure(figsize=(7,4))
    plt.bar(x - width/2, macro_f1, width, label="Macro-F1")
    plt.bar(x + width/2, bal_acc,  width, label="Balanced Acc.")
    plt.xticks(x, nombres_modelos); plt.ylim(0, 1.05)
    for i,(f1v,bav) in enumerate(zip(macro_f1, bal_acc)):
        plt.text(i - width/2, f1v + 0.02, f"{f1v:.3f}", ha="center", fontsize=9)
        plt.text(i + width/2, bav + 0.02, f"{bav:.3f}", ha="center", fontsize=9)
    plt.title("Comparación de métricas por modelo")
    plt.legend(loc="lower right"); plt.tight_layout(); plt.show()

def graficar_importancias_atributos(modelo_rf, nombres_features, top_k=15, titulo="Importancia de atributos (RF)"):
    if not hasattr(modelo_rf, "feature_importances_"): return
    importancias = pd.Series(modelo_rf.feature_importances_, index=nombres_features)
    top = importancias.sort_values(ascending=False).head(top_k).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.35*len(top))))
    plt.barh(range(len(top)), top.values)
    plt.yticks(range(len(top)), top.index)
    plt.xlabel("Importance (Gini)")
    plt.title(titulo)
    for i, v in enumerate(top.values):
        plt.text(v + max(top.values)*0.01, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout(); plt.show()
