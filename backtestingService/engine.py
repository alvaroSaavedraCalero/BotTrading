# backtestingService/engine.py

import logging
import pandas as pd
from typing import List, Tuple, Any, Optional

# Importar librerías TA al nivel superior
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange # Importar ATR aquí también
from pandas import Timestamp

from config.config import INITIAL_CAPITAL, RISK_PERCENT, RR_RATIO, COMMISSION_PER_TRADE, SIMULATED_SPREAD
import binanceService.api as binance_api

# Definir un tipo más específico para la estructura de un trade para claridad
# (Tipo, Precio Entrada, Nivel Salida, Fecha Entrada, Fecha Salida)
TradeTuple = Tuple[str, float, float, Timestamp, Timestamp]


# Backtesting y simulación de resultados financieros
def realizar_backtest(df: pd.DataFrame, model: Any) -> Tuple[float, List[TradeTuple], List[float]]:
    """
    Realiza un backtest simple basado en señales de un modelo y reglas de gestión de riesgo.

    Args:
        df: DataFrame de pandas con datos OHLCV y columnas de características requeridas.
            Se espera que el índice sea de tipo Timestamp.
        model: Modelo de ML entrenado con un método predict(features).
               Se usa Any ya que el tipo exacto puede variar (RandomForest, XGBoost, etc.).

    Returns:
        Una tupla conteniendo:
        - balance final (float)
        - lista de trades ejecutados (List[TradeTuple])
        - historial del balance (List[float])
    """
    # Quitar imports internos, ya están arriba
    # from ta.momentum import RSIIndicator
    # from ta.trend import MACD, EMAIndicator
    # from ta.volatility import AverageTrueRange

    logging.info("Realizando backtesting y simulación de resultados mejorada...")

    # Asegurarse de trabajar con una copia para no modificar el df original fuera de la función
    df_processed = df.copy()

    # --- Cálculo de Indicadores ---
    # Añadir tipos a las columnas ayuda a la claridad, aunque pandas trabaja dinámicamente
    df_processed['returns'] = df_processed['close'].pct_change()
    df_processed['rsi'] = RSIIndicator(df_processed['close'], window=14).rsi()
    df_processed['macd'] = MACD(df_processed['close']).macd_diff()
    df_processed['ema9'] = EMAIndicator(df_processed['close'], window=9).ema_indicator()
    df_processed['ema21'] = EMAIndicator(df_processed['close'], window=21).ema_indicator()
    df_processed['atr'] = AverageTrueRange(df_processed['high'], df_processed['low'], df_processed['close'], window=14).average_true_range()

    # Eliminar filas con NaNs resultantes de los cálculos de indicadores/returns
    df_processed = df_processed.dropna()

    if df_processed.empty:
        logging.warning("DataFrame vacío después de calcular indicadores y eliminar NaNs. No se puede realizar backtest.")
        return INITIAL_CAPITAL, [], [INITIAL_CAPITAL] # Retornar valores iniciales/vacíos

    # --- Inicialización del Backtest ---
    balance: float = INITIAL_CAPITAL
    # Usar float para balance_hist ya que balance es float
    balance_hist: List[float] = [balance]
    trades: List[TradeTuple] = []
    # Definir las columnas de features explícitamente
    feature_cols: List[str] = ['open', 'high', 'low', 'close', 'temporalidad', 'returns', 'rsi', 'macd', 'ema9', 'ema21', 'atr']

    # --- Bucle Principal del Backtest ---
    for i in range(len(df_processed) - 1):
        riesgo_actual: float = balance * RISK_PERCENT
        # Asegurarse que las features existan y manejar posibles NaNs residuales si iloc falla
        try:
            # Extraer la fila actual como DataFrame para mantener estructura de entrada al modelo
            current_features: pd.DataFrame = df_processed[feature_cols].iloc[i:i+1].fillna(0)
            if current_features.empty:
                logging.warning(f"No se pudieron extraer features en el índice {i}. Saltando iteración.")
                balance_hist.append(balance) # Mantener historial de balance
                continue
            prediccion: int = model.predict(current_features)[0]
        except Exception as e:
            logging.error(f"Error al predecir en índice {i}: {e}. Saltando iteración.")
            balance_hist.append(balance) # Mantener historial de balance
            continue

        fecha_entrada: Timestamp = df_processed.index[i]

        # Pasar solo el dataframe hasta la fila actual para la confirmación
        sub_df: pd.DataFrame = df_processed.iloc[:i+1]
        confirmar_long: bool
        confirmar_short: bool
        # Añadir manejo de errores por si confirmar_senal falla
        try:
             confirmar_long, confirmar_short = confirmar_senal_con_indicadores(sub_df)
        except Exception as e:
             logging.error(f"Error al confirmar señal en índice {i}: {e}. Saltando iteración.")
             balance_hist.append(balance)
             continue


        # --- Lógica de Entrada y Salida de Trades ---
        precio_entrada: float = 0.0 # Inicializar
        stop_loss: float = 0.0
        take_profit: float = 0.0
        trade_executed: bool = False # Flag para evitar añadir balance si no hubo trade

        # Lógica para LONG
        if prediccion == 2 and confirmar_long:
            precio_entrada = df_processed['close'].iloc[i] * (1 + SIMULATED_SPREAD)
            # Calcular SL/TP basados en el riesgo porcentual sobre el precio de entrada
            # El cálculo original de SL usaba RISK_PERCENT directamente, lo cual es inusual.
            # Un SL común se basa en ATR o un % fijo *diferente* al riesgo por trade.
            # Asumiendo que RISK_PERCENT define cuánto bajar/subir para SL/TP desde entrada:
            sl_distance: float = precio_entrada * RISK_PERCENT # Distancia absoluta del SL
            tp_distance: float = sl_distance * RR_RATIO      # Distancia absoluta del TP
            stop_loss = precio_entrada - sl_distance
            take_profit = precio_entrada + tp_distance

            for j in range(i + 1, len(df_processed)):
                fecha_salida: Timestamp = df_processed.index[j]
                low_price: float = df_processed['low'].iloc[j]
                high_price: float = df_processed['high'].iloc[j]

                if low_price <= stop_loss: # Salida por Stop Loss
                    # Calcular pérdida basada en el riesgo definido, no en el tamaño del movimiento SL necesariamente
                    balance -= riesgo_actual # Restar el riesgo definido
                    balance -= (riesgo_actual / (precio_entrada - stop_loss)) * precio_entrada * COMMISSION_PER_TRADE * 2 # Comisión estimada sobre nocional
                    trades.append(('LONG_PERDIDA', precio_entrada, stop_loss, fecha_entrada, fecha_salida))
                    trade_executed = True
                    break
                elif high_price >= take_profit: # Salida por Take Profit
                    # Calcular ganancia basada en el riesgo y RR ratio
                    profit: float = riesgo_actual * RR_RATIO
                    balance += profit
                    balance -= (profit / (take_profit - precio_entrada)) * precio_entrada * COMMISSION_PER_TRADE * 2 # Comisión estimada sobre nocional
                    trades.append(('LONG_GANANCIA', precio_entrada, take_profit, fecha_entrada, fecha_salida))
                    trade_executed = True
                    break
            # Si el bucle termina sin break (fin de datos), no se cierra la operación (podría añadirse lógica)

        # Lógica para SHORT
        elif prediccion == 0 and confirmar_short:
            precio_entrada = df_processed['close'].iloc[i] * (1 - SIMULATED_SPREAD)
            sl_distance = precio_entrada * RISK_PERCENT # Distancia absoluta del SL (hacia arriba)
            tp_distance = sl_distance * RR_RATIO      # Distancia absoluta del TP (hacia abajo)
            stop_loss = precio_entrada + sl_distance
            take_profit = precio_entrada - tp_distance

            for j in range(i + 1, len(df_processed)):
                fecha_salida = df_processed.index[j]
                low_price = df_processed['low'].iloc[j]
                high_price = df_processed['high'].iloc[j]

                if high_price >= stop_loss: # Salida por Stop Loss
                    balance -= riesgo_actual
                    balance -= (riesgo_actual / (stop_loss - precio_entrada)) * precio_entrada * COMMISSION_PER_TRADE * 2 # Comisión estimada
                    trades.append(('SHORT_PERDIDA', precio_entrada, stop_loss, fecha_entrada, fecha_salida))
                    trade_executed = True
                    break
                elif low_price <= take_profit: # Salida por Take Profit
                    profit = riesgo_actual * RR_RATIO
                    balance += profit
                    balance -= (profit / (precio_entrada - take_profit)) * precio_entrada * COMMISSION_PER_TRADE * 2 # Comisión estimada
                    trades.append(('SHORT_GANANCIA', precio_entrada, take_profit, fecha_entrada, fecha_salida))
                    trade_executed = True
                    break
            # Si el bucle termina sin break (fin de datos), no se cierra la operación

        # Actualizar historial de balance solo si cambió (hubo trade) o al final
        # El código original lo añadía en cada iteración, ajustamos ligeramente
        # if trade_executed: # Opcional: añadir solo si hubo trade o siempre como antes
        balance_hist.append(balance) # Mantener como antes: añadir en cada paso i

    # Asegurar que el historial tenga la misma longitud que los pasos + inicial
    # if len(balance_hist) == len(df_processed): # El bucle va hasta len-1
    #      balance_hist.append(balance) # Añadir el último estado si es necesario

    logging.info(f"Backtesting finalizado. Balance final: {balance:.2f}")
    return balance, trades, balance_hist



# Función para confirmar la señal con indicadores técnicos
def confirmar_senal_con_indicadores(df: pd.DataFrame) -> Tuple[bool, bool]:
    """
    Confirma señales basadas en la última fila de indicadores calculados sobre el DataFrame.

    Args:
        df: DataFrame con datos OHLC hasta el punto actual. Debe tener suficientes
            filas para calcular los indicadores.

    Returns:
        Una tupla (confirmar_long, confirmar_short) de booleanos.
    """
    # Trabajar con copia para evitar SettingWithCopyWarning
    df_copy = df.copy()

    # Recalcular indicadores sobre el dataframe recibido (hasta el punto actual)
    df_copy['rsi'] = RSIIndicator(df_copy['close'], window=14).rsi()
    macd = MACD(df_copy['close'])
    df_copy['macd_diff'] = macd.macd_diff()
    df_copy['ema_fast'] = EMAIndicator(df_copy['close'], window=9).ema_indicator()
    df_copy['ema_slow'] = EMAIndicator(df_copy['close'], window=21).ema_indicator()

    # Comprobar si hay suficientes datos para obtener el último valor
    if df_copy.empty or df_copy[['rsi', 'macd_diff', 'ema_fast', 'ema_slow']].iloc[-1].isna().any():
        logging.warning("No hay suficientes datos o hay NaNs en los indicadores para confirmar señal.")
        return False, False # No confirmar si faltan datos

    # Obtener el último valor de los indicadores
    rsi: float = df_copy['rsi'].iloc[-1]
    macd_val: float = df_copy['macd_diff'].iloc[-1]
    ema_fast: float = df_copy['ema_fast'].iloc[-1]
    ema_slow: float = df_copy['ema_slow'].iloc[-1]

    # Definir condiciones de confirmación
    confirmar_long: bool = rsi > 50 and macd_val > 0 and ema_fast > ema_slow
    confirmar_short: bool = rsi < 50 and macd_val < 0 and ema_fast < ema_slow

    return confirmar_long, confirmar_short


# Función para calcular tamaño de la orden basado en el riesgo y balance actual
def calcular_cantidad(riesgo: float, stop_loss: float, precio_entrada: float) -> Optional[float]:
    """
    Calcula la cantidad de la orden basada en el riesgo porcentual sobre el saldo
    disponible en USDT y la distancia al stop loss.

    Args:
        riesgo: Riesgo porcentual a asumir sobre el saldo (ej. 0.01 para 1%).
        stop_loss: Precio del stop loss.
        precio_entrada: Precio de entrada de la operación.

    Returns:
        La cantidad calculada redondeada, o None si ocurre un error o no hay saldo.
    """
    try:
        # Asumiendo que consultar_saldo_especifico devuelve Optional[float]
        saldo_usdt: Optional[float]
        # El segundo valor retornado no se usa, se puede ignorar con _
        saldo_usdt, _ = binance_api.consultar_saldo_especifico("USDT") # Balance en USDT

        if saldo_usdt is None or saldo_usdt <= 0:
            logging.warning("⚠️ No hay saldo disponible en USDT o error al consultarlo.")
            return None

        riesgo_monetario: float = saldo_usdt * riesgo # Riesgo total en USDT
        distancia_sl: float = abs(precio_entrada - stop_loss)

        if distancia_sl == 0:
             logging.warning("⚠️ Distancia al Stop Loss es cero. No se puede calcular cantidad.")
             return None

        cantidad: float = riesgo_monetario / distancia_sl
        # Redondeo (precisión puede depender del par, 6 es un ejemplo)
        return round(cantidad, 6)
    except Exception as e:
        logging.error(f"❌ Error al calcular cantidad: {e}", exc_info=True) # Log con traceback
        return None
