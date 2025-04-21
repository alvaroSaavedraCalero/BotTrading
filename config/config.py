# config/config.py

import os, logging, sys # Importar sys para salir del programa si faltan variables
from typing import Dict, Any, Optional # Para tipos de retorno en las funciones
from dotenv import load_dotenv # Para cargar variables de entorno desde .env
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

# --- Machine Learning Model Parameters ---
N_ESTIMATORS: int = 5000
RANDOM_STATE: int = 500
MODEL_CONFIDENCE_THRESHOLD: float = 0.60  # Minimum confidence for signal generation

# --- Technical Indicator Parameters ---
VOLATILITY_WINDOW: int = 20       # Window for volatility calculations
MOMENTUM_WINDOW: int = 14         # Window for momentum indicators
VOLUME_WINDOW: int = 20          # Window for volume analysis
TREND_WINDOW: int = 50           # Window for trend indicators

# Parámetros de backtesting
INITIAL_CAPITAL: float = 100.0       # Usar float para capital es más flexible
RISK_PERCENT: float = 0.01
RR_RATIO: float = 2.0                # Usar float para ratios

# Parámetros de operación
COMMISSION_PER_TRADE: float = 0.001  # 0.1%
SIMULATED_SPREAD: float = 0.0005      # 0.05%

# Límites de tamaño de posición como porcentaje del capital
MIN_POSITION_SIZE: float = 0.01      # Mínimo 1% del capital
MAX_POSITION_SIZE: float = 0.20      # Máximo 20% del capital

# --- Risk Management Parameters ---
RISK_PARAMS = {
    'ATR_WINDOW': 14,            # Window for ATR calculation
    'ATR_MULTIPLIER': 1.5,       # Multiplier for ATR-based stops
    'MIN_RISK_PERCENT': 0.005,   # Minimum risk per trade (0.5%)
    'MAX_RISK_PERCENT': 0.02,    # Maximum risk per trade (2%)
    'BASE_RISK_PERCENT': 0.01,   # Base risk percentage (1%)
    'RR_RATIO': 2.0,            # Risk-Reward ratio
    'VOLUME_FACTOR': 1.0,        # Volume impact on position sizing
    'TREND_STRENGTH_FACTOR': 0.5 # Trend strength impact on position sizing
}

# --- Position Management Parameters ---
POSITION_PARAMS = {
    'MAX_POSITIONS': 3,          # Maximum number of simultaneous positions
    'MAX_CORRELATION': 0.7,      # Maximum correlation between simultaneous positions
    'PARTIAL_TAKE_PROFIT': {     # Partial take-profit levels
        'LEVEL_1': {'percent': 0.3, 'at_price': 1.5},  # Close 30% at 1.5x risk
        'LEVEL_2': {'percent': 0.3, 'at_price': 2.0}   # Close 30% at 2x risk
    },
    'TRAILING_STOP': {
        'ACTIVATION': 1.5,       # Activate at 1.5x risk
        'STEP': 0.5             # Trail by 0.5x ATR
    }
}

# --- Feature Engineering Parameters ---
FEATURE_PARAMS = {
    'VOLATILITY': {
        'bb_window': VOLATILITY_WINDOW,
        'bb_dev': 2.0,
        'atr_window': 14
    },
    'MOMENTUM': {
        'rsi_window': MOMENTUM_WINDOW,
        'stoch_window': 14,
        'stoch_smooth': 3,
        'roc_window': 12
    },
    'TREND': {
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'adx_window': 14
    },
    'VOLUME': {
        'obv_ma_window': VOLUME_WINDOW,
        'vwap_window': 14
    }
}

# --- Signal Confirmation Parameters ---
CONFIRMATION_PARAMS = {
    'MIN_ADX': 25,              # Minimum ADX for trend confirmation
    'MIN_VOLUME_RATIO': 1.2,    # Minimum volume ratio for confirmation
    'RSI_OVERSOLD': 30,         # RSI oversold level
    'RSI_OVERBOUGHT': 70,       # RSI overbought level
    'MACD_THRESHOLD': 0,        # MACD threshold for trend confirmation
    'BB_THRESHOLD': 0.8         # Bollinger Band threshold for volatility confirmation
}

# --- Performance Monitoring Parameters ---
MONITORING_PARAMS = {
    'MAX_DRAWDOWN': 0.15,       # Maximum allowed drawdown (15%)
    'MIN_WIN_RATE': 0.40,       # Minimum required win rate
    'MIN_PROFIT_FACTOR': 1.5,   # Minimum required profit factor
    'EVALUATION_PERIOD': 30,    # Days for rolling performance evaluation
    'RECALIBRATION_THRESHOLD': 0.2  # Performance drop triggering recalibration
}

# --- Directory Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Directorio raíz del proyecto (BotPrincipalPython)
MODEL_SAVE_DIR: str = os.path.join(BASE_DIR, "data", "models") # Crea una carpeta 'data/models'
REPORTS_DIR: str = os.path.join(BASE_DIR, "data", "reports") # Crea una carpeta 'data/reports'
PERFORMANCE_DIR: str = os.path.join(BASE_DIR, "data", "performance") # Para métricas de rendimiento
BACKTEST_DIR: str = os.path.join(BASE_DIR, "data", "backtest") # Para resultados de backtesting

# Crear directorios si no existen
for directory in [MODEL_SAVE_DIR, REPORTS_DIR, PERFORMANCE_DIR, BACKTEST_DIR]:
    os.makedirs(directory, exist_ok=True)
# Añadir esas carpetas al .gitignore para evitar subir modelos y reportes innecesarios

# --- Consideración Adicional ---
# Recuerda que sería buena práctica verificar que las variables Optional[str]
# (BINANCE_API_KEY, etc.) no sean None antes de usarlas en otras partes
# del código si son críticas para la ejecución.
def get_risk_parameters() -> Dict[str, float]:
    """
    Get risk management parameters.
    
    Returns:
        Dictionary with risk management parameters
    """
    return {
        'atr_multiple': float(RISK_PARAMS['ATR_MULTIPLIER']),
        'min_risk_percent': float(RISK_PARAMS['MIN_RISK_PERCENT']),
        'max_risk_percent': float(RISK_PARAMS['MAX_RISK_PERCENT']),
        'base_risk_percent': float(RISK_PARAMS['BASE_RISK_PERCENT']),
        'rr_ratio': float(RISK_PARAMS['RR_RATIO']),
        'volume_factor': float(RISK_PARAMS['VOLUME_FACTOR']),
        'trend_strength_factor': float(RISK_PARAMS['TREND_STRENGTH_FACTOR'])
    }

def get_position_parameters() -> Dict[str, Any]:
    """
    Get position management parameters.
    
    Returns:
        Dictionary with position management parameters
    """
    return POSITION_PARAMS

def get_feature_parameters() -> Dict[str, Dict[str, Any]]:
    """
    Get feature engineering parameters.
    
    Returns:
        Dictionary with feature engineering parameters
    """
    return FEATURE_PARAMS

def get_confirmation_parameters() -> Dict[str, float]:
    """
    Get signal confirmation parameters.
    
    Returns:
        Dictionary with signal confirmation parameters
    """
    return {k: float(v) for k, v in CONFIRMATION_PARAMS.items()}

def get_monitoring_parameters() -> Dict[str, float]:
    """
    Get performance monitoring parameters.
    
    Returns:
        Dictionary with monitoring parameters
    """
    return {k: float(v) for k, v in MONITORING_PARAMS.items()}

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