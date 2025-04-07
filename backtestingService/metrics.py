# backtestingService/metrics.py


import numpy as np
# Importar tipos necesarios
from typing import List, Tuple, Dict, Any
# Importar Timestamp si el índice del DataFrame es de ese tipo y se usa en TradeTuple
# Si no, se podría usar datetime.datetime o Any
from pandas import Timestamp # O from datetime import datetime

# Definir el tipo TradeTuple como se espera que llegue desde engine.py
# (Tipo_Resultado, Precio_Entrada, Nivel_Salida, Fecha_Entrada, Fecha_Salida)
# Asumiendo que las fechas son Timestamps de pandas
TradeTuple = Tuple[str, float, float, Timestamp, Timestamp]


# Función para calcular las métricas de rendimiento
def calcular_metricas(balance_hist: List[float], trades: List[TradeTuple]) -> Dict[str, float]:
    """
    Calcula métricas de rendimiento clave a partir del historial de balance y trades.

    Args:
        balance_hist: Lista del valor del balance en cada paso del backtest.
        trades: Lista de tuplas, donde cada tupla representa un trade ejecutado.
                Se asume la estructura TradeTuple definida arriba.

    Returns:
        Un diccionario con el nombre de la métrica y su valor calculado.
    """
    if not balance_hist: # Comprobar si el historial de balance está vacío
        return {
            "Profit Factor": 0.0,
            "Max Drawdown": 0.0,
            "Sharpe Ratio": 0.0
        }

    # Filtrar trades ganadores y perdedores
    ganancias: List[TradeTuple] = [t for t in trades if "GANANCIA" in t[0]]
    perdidas: List[TradeTuple] = [t for t in trades if "PERDIDA" in t[0]]

    # --- Profit Factor ---
    # !! Advertencia: La lógica original suma t[2] (nivel de salida, SL/TP).
    # Esto probablemente NO es el beneficio/pérdida real del trade.
    # Para un Profit Factor correcto, necesitarías calcular el P/L de cada trade.
    # Mantendremos la lógica original aquí, pero DEBE REVISARSE.
    total_ganancias_calculado: float = sum([t[2] for t in ganancias]) if ganancias else 0.0
    total_perdidas_calculado: float = sum([t[2] for t in perdidas]) if perdidas else 0.0 # No usar 1 como default
    
    profit_factor: float
    if total_perdidas_calculado != 0:
        # Usar abs() por si los niveles de salida de pérdidas son negativos (no deberían serlo)
        profit_factor = total_ganancias_calculado / abs(total_perdidas_calculado)
    elif total_ganancias_calculado > 0: # Ganancias pero no pérdidas
        profit_factor = np.inf
    else: # Ni ganancias ni pérdidas (o ambos 0)
        profit_factor = 0.0

    # --- Maximum Drawdown ---
    peak: float = balance_hist[0]
    max_drawdown: float = 0.0
    for balance in balance_hist:
        # Asegurarse que balance es float
        current_balance: float = float(balance)
        peak = max(peak, current_balance)
        # Evitar división por cero si el peak es 0
        drawdown: float = (peak - current_balance) / peak if peak != 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)

    # --- Sharpe Ratio ---
    # Convertir balance_hist a numpy array para operaciones vectorizadas
    balance_array: np.ndarray = np.array(balance_hist, dtype=float)

    sharpe_ratio: float = 0.0
    if len(balance_array) > 1:
        # Calcular retornos. Nota: np.diff(a) / a[:-1] son retornos simples.
        returns: np.ndarray = np.diff(balance_array) / balance_array[:-1]

        # Calcular desviación estándar y convertirla explícitamente a float
        std_dev_returns: float = float(np.std(returns)) # <--- Añadido float()

        if std_dev_returns != 0:
            # Calcular Sharpe Ratio y convertirlo explícitamente a float
            sharpe_ratio = float(np.mean(returns) / std_dev_returns) # <--- Añadido float()
        # Podría anualizarse multiplicando por sqrt(N)...

    return {
        "Profit Factor": round(profit_factor, 2),
        "Max Drawdown": round(max_drawdown * 100, 2), # En porcentaje
        "Sharpe Ratio": round(sharpe_ratio, 2)
    }