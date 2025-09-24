from dataclasses import dataclass

@dataclass(frozen=True)
class ConfiguracionSimulacion:
    minutos_por_partido: int = 90
    cantidad_temporadas: int = 10000   
    semilla_aleatoria: int = 42
    truncamiento_poisson: int = 12    
