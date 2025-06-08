import sys
from pathlib import Path
import pandas as pd

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtestingService.metrics import calcular_metricas


def test_calcular_metricas_normal_case():
    balance_hist = [1000, 1050, 1030, 1060]
    trades = [
        ('LONG_GANANCIA', 100, 110, pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')),
        ('LONG_PERDIDA', 120, 110, pd.Timestamp('2024-01-03'), pd.Timestamp('2024-01-04')),
        ('SHORT_GANANCIA', 150, 130, pd.Timestamp('2024-01-05'), pd.Timestamp('2024-01-06')),
    ]

    metrics = calcular_metricas(balance_hist, trades)
    assert metrics == {
        'Profit Factor': 2.18,
        'Max Drawdown': 1.9,
        'Sharpe Ratio': 0.69,
    }


def test_calcular_metricas_empty_inputs():
    metrics = calcular_metricas([], [])
    assert metrics == {
        'Profit Factor': 0.0,
        'Max Drawdown': 0.0,
        'Sharpe Ratio': 0.0,
    }
