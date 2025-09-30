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
    """Devuelve una muestra del test estratificada por 'label' (fallback aleatorio si no es posible)."""
    tamano = min(tamano, len(test_df))
    try:
        from sklearn.model_selection import train_test_split
        # train_size = tamano -> devolvemos 'sample' con ese tamaño
        sample, _ = train_test_split(
            test_df, train_size=tamano, stratify=test_df["label"], random_state=semilla
        )
        return sample
    except Exception:
        return test_df.sample(n=tamano, random_state=semilla) 
