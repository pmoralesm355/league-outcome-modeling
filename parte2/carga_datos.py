import io, zipfile, urllib.request
import pandas as pd
from configuracion import URL_POKER_ZIP

COLUMNAS = [f"{p}{i}" for i in range(1, 6) for p in ("S", "R")] + ["label"]

def cargar_dataset_poker(url: str = URL_POKER_ZIP):
    with urllib.request.urlopen(url) as resp:
        zbytes = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(zbytes))
    with zf.open("poker-hand-training-true.data") as f:
        train = pd.read_csv(f, header=None)
    with zf.open("poker-hand-testing.data") as f:
        test = pd.read_csv(f, header=None)
    train.columns = COLUMNAS
    test.columns = COLUMNAS
    return train, test

def muestrear_test(test_df: pd.DataFrame, tamano: int, semilla: int | None = None) -> pd.DataFrame:
    """Devuelve una muestra del test estratificada por 'label' (y, si es posible, con ≥1 instancia por clase)."""
    tamano = min(tamano, len(test_df))
    if tamano <= 0:
        return test_df.head(0).copy()

    labels_col = "label"
    clases = test_df[labels_col].unique()

    # (tamaño) garantiza al menos 1 por clase
    if tamano >= len(clases):
        base = test_df.groupby(labels_col, group_keys=False).head(1)
        resto = test_df.drop(base.index)
        faltan = tamano - len(base)

        if faltan > 0:
            try:
                from sklearn.model_selection import train_test_split
                extra, _ = train_test_split(
                    resto, train_size=faltan,
                    stratify=resto[labels_col], random_state=semilla
                )
            except Exception:
                extra = resto.sample(n=faltan, random_state=semilla)
            muestra = pd.concat([base, extra], ignore_index=False)
        else:
            muestra = base

        return muestra.sample(frac=1, random_state=semilla).reset_index(drop=True)

    # (no suf tamaño) estratifica; si falla, aleatorio
    try:
        from sklearn.model_selection import train_test_split
        sample, _ = train_test_split(
            test_df, train_size=tamano,
            stratify=test_df[labels_col], random_state=semilla
        )
        return sample.reset_index(drop=True)
    except Exception:
        return test_df.sample(n=tamano, random_state=semilla).reset_index(drop=True)