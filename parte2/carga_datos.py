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
    tamano = min(tamano, len(test_df))
    return test_df.sample(n=tamano, random_state=semilla)
