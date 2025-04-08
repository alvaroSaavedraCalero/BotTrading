# config/config.py

import os
from dotenv import load_dotenv
from typing import Optional # Importar Optional para los resultados de getenv

# Cargar variables de entorno desde .env
# Añadimos tipo al path por claridad
dotenv_path: str = os.path.join(os.path.dirname(__file__), "..", "var.env")
load_dotenv(dotenv_path=dotenv_path)

# Claves API (os.getenv devuelve Optional[str])
# La aplicación deberá manejar el caso en que sean None si son obligatorias
BINANCE_API_KEY: Optional[str] = os.getenv("API_KEY")
BINANCE_API_SECRET: Optional[str] = os.getenv("API_SECRET")

TELEGRAM_TOKEN: Optional[str] = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID") # Los IDs suelen manejarse mejor como string

# Parámetros del modelo
N_ESTIMATORS: int = 5000
RANDOM_STATE: int = 500

# Parámetros de backtesting
INITIAL_CAPITAL: float = 100.0       # Usar float para capital es más flexible
RISK_PERCENT: float = 0.01
RR_RATIO: float = 2.0                # Usar float para ratios

# Parámetros de operación
COMMISSION_PER_TRADE: float = 0.001  # 0.1%
SIMULATED_SPREAD: float = 0.0005      # 0.05%

# Umbral para definir target en clasificación
TARGET_THRESHOLD: float = 0.001

# Directorios por defecto (os.path.expanduser devuelve str)
MODEL_SAVE_DIR: str = os.path.expanduser("~/OneDrive/Escritorio/InfoRecursosBots/ModelosEntrenados")
REPORTS_DIR: str = os.path.expanduser("~/OneDrive/Escritorio/InfoRecursosBots/ResultadosBots")

# --- Consideración Adicional ---
# Recuerda que sería buena práctica verificar que las variables Optional[str]
# (BINANCE_API_KEY, etc.) no sean None antes de usarlas en otras partes
# del código si son críticas para la ejecución.
#