import numpy as np
from pathlib import Path

def crear_generador_aleatorio(semilla: int) -> np.random.Generator:
    """Crea un generador de números aleatorios reproducible."""
    return np.random.default_rng(semilla)

def asegurar_directorio(ruta: str | Path) -> Path:
    """Crea un directorio si no existe y devuelve su Path."""
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta
