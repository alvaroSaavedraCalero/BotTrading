# config/config.py

import os, logging, sys # Importar sys para salir del programa si faltan variables
from dotenv import load_dotenv
from typing import Optional # Importar Optional para los resultados de getenv

# Cargar variables de entorno desde .env
# Añadimos tipo al path por claridad
dotenv_path: str = os.path.join(os.path.dirname(__file__), "..", ".env")
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Directorio raíz del proyecto (BotPrincipalPython)
MODEL_SAVE_DIR: str = os.path.join(BASE_DIR, "data", "models") # Crea una carpeta 'data/models'
REPORTS_DIR: str = os.path.join(BASE_DIR, "data", "reports") # Crea una carpeta 'data/reports'

# Añadir esas carpetas al .gitignore para evitar subir modelos y reportes innecesarios

# --- Consideración Adicional ---
# Recuerda que sería buena práctica verificar que las variables Optional[str]
# (BINANCE_API_KEY, etc.) no sean None antes de usarlas en otras partes
# del código si son críticas para la ejecución.
#

def comprobar_variables_entorno():
    """
    Comprueba si las variables de entorno necesarias están definidas.

    Lanza un error si alguna variable de entorno crítica no está definida.
    """
    # Al inicio de main.py (o en una función de validación llamada desde config o main)
    required_env_vars = ["API_KEY", "API_SECRET", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logging.error(f"Faltan variables de entorno esenciales en '.env': {', '.join(missing_vars)}")
        sys.exit(1) # Salir si faltan variables críticas