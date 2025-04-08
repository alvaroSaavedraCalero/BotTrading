# utils/enumerados.py

from enum import Enum, auto # Importar auto es una buena práctica opcional

class TargetMethod(Enum):
    """Enumeración de los métodos disponibles para definir el target en el entrenamiento."""
    # Quitar comas al final para que el .value sea int, no Tuple[int]
    ORIGINAL = 1
    ATR = 2
    HORIZONTE_N = 3
    NIVEL_ALCANZADO = 4

    # Alternativa si los valores numéricos específicos no importan:
    # ORIGINAL = auto()
    # ATR = auto()
    # HORIZONTE_N = auto()
    # NIVEL_ALCANZADO = auto()


class Modelo(Enum):
    """Enumeración de los tipos de modelos de Machine Learning soportados."""
    GRADIENT_BOOSTING = 1
    RANDOM_FOREST = 2
    XGB = 3

    # Alternativa si los valores numéricos específicos no importan:
    # GRADIENT_BOOSTING = auto()
    # RANDOM_FOREST = auto()
    # XGB = auto()