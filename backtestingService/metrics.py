
import numpy as np


# Función para calcular las métricas de rendimiento
def calcular_metricas(balance_hist, trades):
    ganancias = [t for t in trades if "GANANCIA" in t[0]]
    perdidas = [t for t in trades if "PERDIDA" in t[0]]
    total_ganancias = sum([t[2] for t in ganancias]) if ganancias else 0
    total_perdidas = sum([t[2] for t in perdidas]) if perdidas else 1  # Evitar división por 0
    profit_factor = total_ganancias / abs(total_perdidas) if total_perdidas != 0 else np.inf
    # Maximum Drawdown (máxima caída del balance)
    peak = balance_hist[0]
    max_drawdown = 0
    for balance in balance_hist:
        peak = max(peak, balance)
        drawdown = (peak - balance) / peak
        max_drawdown = max(max_drawdown, drawdown)
    # Sharpe Ratio (rentabilidad ajustada al riesgo)
    returns = np.diff(balance_hist) / balance_hist[:-1]  # Retornos logarítmicos
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
    return {
        "Profit Factor": round(profit_factor, 2),
        "Max Drawdown": round(max_drawdown * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2)
    }