import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.trainer import entrenar_modelo
from utils.enumerados import TargetMethod


def test_entrenar_modelo_runs():
    rows = 80
    data = {
        'open': pd.Series(range(rows), dtype=float),
        'high': pd.Series(range(1, rows + 1), dtype=float),
        'low': pd.Series(range(rows), dtype=float),
        'close': pd.Series(range(rows), dtype=float),
        'temporalidad': pd.Series([1] * rows, dtype=float),
    }
    df = pd.DataFrame(data)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    trained = entrenar_modelo(model, df, target_method=TargetMethod.ORIGINAL)
    assert trained is not None

