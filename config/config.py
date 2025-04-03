import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "config", "var.env"))

# Claves API
BINANCE_API_KEY = os.getenv("API_KEY")
BINANCE_API_SECRET = os.getenv("API_SECRET")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Parámetros del modelo
N_ESTIMATORS = 5000
RANDOM_STATE = 500

# Parámetros de backtesting
INITIAL_CAPITAL = 100
RISK_PERCENT = 0.01
RR_RATIO = 2

# Parámetros de operación
COMMISSION_PER_TRADE = 0.001   # 0.1%
SIMULATED_SPREAD = 0.0005      # 0.05%

# Umbral para definir target en clasificación
TARGET_THRESHOLD = 0.001

# Directorios por defecto
MODEL_SAVE_DIR = os.path.expanduser("~/OneDrive/Escritorio/InfoRecursosBots/ModelosEntrenados")
REPORTS_DIR = os.path.expanduser("~/OneDrive/Escritorio/InfoRecursosBots/ResultadosBots")