import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def crear_modelos(y_entrenamiento):
    rf = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1
    )
    hgb = HistGradientBoostingClassifier(
        max_depth=None, learning_rate=0.08, max_iter=300,
        l2_regularization=0.0, random_state=42
    )
    frec = pd.Series(y_entrenamiento).value_counts()
    inv_freq = {cl: 1.0/c for cl, c in frec.items()}
    pesos_muestra = np.array([inv_freq[y] for y in y_entrenamiento], dtype=np.float32)
    return rf, hgb, pesos_muestra

def entrenar_modelos(rf, hgb, X_train, y_train, sample_weight):
    rf.fit(X_train, y_train)
    hgb.fit(X_train, y_train, sample_weight=sample_weight)
    return rf, hgb
