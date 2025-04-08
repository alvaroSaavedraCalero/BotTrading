# core/bot.py

import logging
import pandas as pd # Importar pandas
from typing import Dict, Any, Optional, Tuple, TypedDict, List # Importar tipos

# Importar configuración y otras dependencias (asumiendo tipos de config)
from config.config import RISK_PERCENT, RR_RATIO
from binanceService.api import ejecutar_orden_binance # Asume que esta función está tipada (retorna None)
from backtestingService.engine import confirmar_senal_con_indicadores # Asume que esta función está tipada (retorna Tuple[bool, bool])

# Definir la estructura del diccionario de señal usando TypedDict para mayor claridad
class SignalDict(TypedDict):
    """Estructura del diccionario retornado por generar_senal_tiempo_real."""
    Tipo: str
    Precio_Entrada: Optional[float] # Usar _ en lugar de espacio para compatibilidad
    Stop_Loss: Optional[float]
    Take_Profit: Optional[float]


# Generar señal claramente estructurada
def generar_senal_tiempo_real(
    model: Any, # El modelo entrenado (tipo específico desconocido, Any es flexible)
    df: pd.DataFrame, # DataFrame con los datos más recientes (OHLC + indicadores necesarios)
    simbolo: str,
    modo_simulacion: bool = True
) -> SignalDict: # Retorna el diccionario tipado
    """
    Genera una señal de trading basada en la predicción del modelo y la confirmación
    de indicadores técnicos. Opcionalmente ejecuta la orden si no está en modo simulación.

    Args:
        model: El modelo de ML entrenado.
        df: DataFrame con los datos de mercado más recientes necesarios para
            la predicción y la confirmación de indicadores.
        simbolo: El símbolo del par (ej. 'BTCUSDT').
        modo_simulacion: Si es True, solo loggea la señal; si es False, intenta ejecutarla.

    Returns:
        Un diccionario (SignalDict) representando la señal generada (LONG, SHORT o NEUTRAL).
    """
    try:
        # --- Predicción del Modelo ---
        # !! ADVERTENCIA: Asegúrate de que las features usadas aquí coincidan
        # con las usadas durante el ENTRENAMIENTO del modelo en trainer.py !!
        # El código original solo usa OHLC, pero trainer.py usaba más (rsi, macd, etc.)
        # Si el modelo se entrenó con más features, DEBES incluirlas aquí.
        # Ejemplo (si se necesitan todas las features):
        # features_cols: List[str] = ['open', 'high', 'low', 'close', 'temporalidad', 'returns', 'rsi', 'macd', 'ema9', 'ema21', 'atr']
        # features: pd.DataFrame = df[features_cols].iloc[-1:].fillna(0) # Usar la última fila
        # Por ahora, mantenemos las features originales del código proporcionado:
        features_cols_prediccion: List[str] = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in features_cols_prediccion):
             logging.error(f"Faltan columnas {features_cols_prediccion} en el DataFrame para la predicción.")
             # Retornar señal NEUTRAL por defecto en caso de error de datos
             return SignalDict(Tipo="NEUTRAL ⚪️", Precio_Entrada=None, Stop_Loss=None, Take_Profit=None)

        features: pd.DataFrame = df[features_cols_prediccion].iloc[-1:] # Usar la última fila
        prediccion: int = int(model.predict(features)[0]) # Convertir predicción a int estándar

        # Obtener precio actual del cierre de la última vela
        precio_actual: float = float(df['close'].iloc[-1]) # Convertir a float estándar

        # --- Confirmación con Indicadores ---
        # confirmar_senal_con_indicadores necesita el df con indicadores calculados
        # Asegurarse que df ya los tiene o calcularlos aquí si es necesario.
        # Asumimos que df ya viene preparado por el proceso que llama a esta función.
        confirmar_long: bool
        confirmar_short: bool
        confirmar_long, confirmar_short = confirmar_senal_con_indicadores(df)

        # --- Lógica de Señal y Ejecución ---
        stop_loss: float = 0.0
        take_profit: float = 0.0

        if prediccion == 2 and confirmar_long:
            # Calcular SL/TP
            stop_loss = precio_actual * (1 - RISK_PERCENT)
            take_profit = precio_actual * (1 + RISK_PERCENT * RR_RATIO)
            tipo_senal: str = "LONG 🚀"

            if modo_simulacion:
                logging.info(f"💡 {tipo_senal} SIMULADA {simbolo} | Entrada ~{precio_actual:.4f} | SL: {stop_loss:.4f} | TP: {take_profit:.4f}")
            else:
                logging.info(f"🔥 EJECUTANDO {tipo_senal} {simbolo} | Entrada ~{precio_actual:.4f} | SL: {stop_loss:.4f} | TP: {take_profit:.4f}")
                ejecutar_orden_binance(simbolo, "LONG", precio_actual, stop_loss, take_profit) # tipo="LONG"

            # Retornar diccionario con la estructura definida
            return SignalDict(Tipo=tipo_senal, Precio_Entrada=precio_actual, Stop_Loss=stop_loss, Take_Profit=take_profit)

        elif prediccion == 0 and confirmar_short:
            # Calcular SL/TP
            stop_loss = precio_actual * (1 + RISK_PERCENT)
            take_profit = precio_actual * (1 - RISK_PERCENT * RR_RATIO)
            tipo_senal: str = "SHORT 📉"

            if modo_simulacion:
                logging.info(f"💡 {tipo_senal} SIMULADA {simbolo} | Entrada ~{precio_actual:.4f} | SL: {stop_loss:.4f} | TP: {take_profit:.4f}")
            else:
                logging.info(f"🔥 EJECUTANDO {tipo_senal} {simbolo} | Entrada ~{precio_actual:.4f} | SL: {stop_loss:.4f} | TP: {take_profit:.4f}")
                ejecutar_orden_binance(simbolo, "SHORT", precio_actual, stop_loss, take_profit) # tipo="SHORT"

            # Retornar diccionario con la estructura definida
            return SignalDict(Tipo=tipo_senal, Precio_Entrada=precio_actual, Stop_Loss=stop_loss, Take_Profit=take_profit)

        else: # Si no hay predicción clara (prediccion == 1) o no hay confirmación
             logging.info(f"🔍 Señal ({prediccion=}) NO confirmada por indicadores ({confirmar_long=}, {confirmar_short=}) para {simbolo}.")
             # Retornar diccionario NEUTRAL
             return SignalDict(Tipo="NEUTRAL ⚪️", Precio_Entrada=None, Stop_Loss=None, Take_Profit=None)

    except Exception as e:
        logging.error(f"❌ Error al generar señal en tiempo real para {simbolo}: {e}", exc_info=True)
        # Retornar señal NEUTRAL en caso de cualquier error inesperado
        return SignalDict(Tipo="NEUTRAL ⚪️", Precio_Entrada=None, Stop_Loss=None, Take_Profit=None)