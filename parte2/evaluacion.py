import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, f1_score, balanced_accuracy_score, confusion_matrix
)
from configuracion import TODAS_LAS_CLASES

def evaluar_modelo(modelo, X, y, nombre):
    y_pred = modelo.predict(X)
    f1_macro = f1_score(y, y_pred, labels=TODAS_LAS_CLASES, average="macro", zero_division=0)
    bal_acc  = balanced_accuracy_score(y, y_pred)
    print(f"\n {nombre}")
    print("Balanced Accuracy:", round(bal_acc, 4))
    print("Macro F1:", round(f1_macro, 4))
    f1_por_clase = f1_score(y, y_pred, labels=TODAS_LAS_CLASES, average=None, zero_division=0)
    print("\nF1 por clase:")
    print(pd.Series(f1_por_clase, index=TODAS_LAS_CLASES))
    print("\nReporte completo:")
    print(classification_report(y, y_pred, labels=TODAS_LAS_CLASES, digits=4, zero_division=0))
    cm = confusion_matrix(y, y_pred, labels=TODAS_LAS_CLASES)
    return y_pred, cm, f1_macro, bal_acc

def resumen_metricas_modelos(resultados):
    df = pd.DataFrame(resultados, columns=["modelo","macro_f1","balanced_accuracy"])
    return df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
