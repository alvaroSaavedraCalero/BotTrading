from enum import Enum

class TargetMethod(Enum):
    ORIGINAL = 1,
    ATR = 2,
    HORIZONTE_N = 3,
    NIVEL_ALCANZADO = 4
    
    
class Modelo(Enum):
    GRADIENT_BOOSTING = 1
    RANDOM_FOREST = 2
    XGB = 3