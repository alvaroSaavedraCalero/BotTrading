# backtestingService/engine.py

import logging
import pandas as pd
from typing import List, Tuple, Any, Optional, Dict # Added Dict

# Importar librerías TA al nivel superior
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange # Importar ATR aquí también
from pandas import Timestamp

from config.config import INITIAL_CAPITAL, RISK_PERCENT, RR_RATIO, COMMISSION_PER_TRADE, SIMULATED_SPREAD
import binanceService.api as binance_api

# Definir un tipo más específico para la estructura de un trade para claridad
# (Tipo, Precio Entrada, Nivel Salida, Fecha Entrada, Fecha Salida, PnL Neto)
TradeTuple = Tuple[str, float, float, Timestamp, Timestamp, float]


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
    open_trade_details: Optional[Dict[str, Any]] = None # For tracking open trades
    # Definir las columnas de features explícitamente
    feature_cols: List[str] = ['open', 'high', 'low', 'close', 'returns', 'rsi', 'macd', 'ema9', 'ema21', 'atr']

    # --- Bucle Principal del Backtest ---
    for i in range(len(df_processed) - 1):
        # Reset trade_executed flag for the current iteration
        trade_executed: bool = False # Moved here to reset for each potential new trade
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

        # Extraer valores de indicadores para la fila actual
        current_rsi = df_processed['rsi'].iloc[i]
        current_macd_diff = df_processed['macd'].iloc[i]
        current_ema_fast = df_processed['ema9'].iloc[i]
        current_ema_slow = df_processed['ema21'].iloc[i]

        confirmar_long: bool = False
        confirmar_short: bool = False

        if pd.isna(current_rsi) or pd.isna(current_macd_diff) or pd.isna(current_ema_fast) or pd.isna(current_ema_slow):
            logging.warning(f"NaNs en indicadores en índice {i} ({df_processed.index[i]}). No se puede confirmar señal.")
            confirmar_long, confirmar_short = False, False
        else:
            try:
                confirmar_long, confirmar_short = confirmar_senal_con_indicadores(
                    current_rsi, current_macd_diff, current_ema_fast, current_ema_slow
                )
            except Exception as e:
                 logging.error(f"Error al confirmar señal en índice {i} ({df_processed.index[i]}): {e}. Saltando iteración.")
                 balance_hist.append(balance) # Mantener historial de balance
                 continue


        # --- Lógica de Entrada y Salida de Trades ---
        precio_entrada: float = 0.0 # Inicializar
        stop_loss: float = 0.0
        take_profit: float = 0.0
        # trade_executed: bool = False # Flag para evitar añadir balance si no hubo trade (moved to top of loop)

        # Lógica para LONG
        if prediccion == 2 and confirmar_long and open_trade_details is None: # Only open if no trade is open
            precio_entrada = df_processed['close'].iloc[i] * (1 + SIMULATED_SPREAD)
            sl_distance: float = precio_entrada * RISK_PERCENT
            tp_distance: float = sl_distance * RR_RATIO
            stop_loss = precio_entrada - sl_distance
            take_profit = precio_entrada + tp_distance
            open_trade_details = {'type': 'LONG', 'entry_price': precio_entrada, 'entry_date': fecha_entrada, 'sl': stop_loss, 'tp': take_profit}
            logging.info(f"Opening LONG trade: {open_trade_details}")


            for j in range(i + 1, len(df_processed)):
                fecha_salida: Timestamp = df_processed.index[j]
                low_price: float = df_processed['low'].iloc[j]
                high_price: float = df_processed['high'].iloc[j]

                if low_price <= stop_loss: # Salida por Stop Loss
                    # Calcular pérdida basada en el riesgo definido, no en el tamaño del movimiento SL necesariamente
                    balance -= riesgo_actual # Restar el riesgo definido
                    balance -= (riesgo_actual * COMMISSION_PER_TRADE * 2) # Nueva comisión
                    trade_pnl_net = -riesgo_actual - (riesgo_actual * COMMISSION_PER_TRADE * 2)
                    trades.append(('LONG_PERDIDA', precio_entrada, stop_loss, fecha_entrada, fecha_salida, trade_pnl_net))
                    trade_executed = True
                    open_trade_details = None # Trade closed
                    break
                elif high_price >= take_profit: # Salida por Take Profit
                    # Calcular ganancia basada en el riesgo y RR ratio
                    profit: float = riesgo_actual * RR_RATIO
                    balance += profit
                    balance -= (riesgo_actual * COMMISSION_PER_TRADE * 2) # Nueva comisión
                    trade_pnl_net = profit - (riesgo_actual * COMMISSION_PER_TRADE * 2)
                    trades.append(('LONG_GANANCIA', precio_entrada, take_profit, fecha_entrada, fecha_salida, trade_pnl_net))
                    trade_executed = True
                    open_trade_details = None # Trade closed
                    break
            # Si el bucle termina sin break (fin de datos), no se cierra la operación (podría añadirse lógica)

        # Lógica para SHORT
        elif prediccion == 0 and confirmar_short and open_trade_details is None: # Only open if no trade is open
            precio_entrada = df_processed['close'].iloc[i] * (1 - SIMULATED_SPREAD)
            sl_distance = precio_entrada * RISK_PERCENT
            tp_distance = sl_distance * RR_RATIO
            stop_loss = precio_entrada + sl_distance
            take_profit = precio_entrada - tp_distance
            open_trade_details = {'type': 'SHORT', 'entry_price': precio_entrada, 'entry_date': fecha_entrada, 'sl': stop_loss, 'tp': take_profit}
            logging.info(f"Opening SHORT trade: {open_trade_details}")


            for j in range(i + 1, len(df_processed)):
                fecha_salida = df_processed.index[j]
                low_price = df_processed['low'].iloc[j]
                high_price = df_processed['high'].iloc[j]

                if high_price >= stop_loss: # Salida por Stop Loss
                    balance -= riesgo_actual
                    balance -= (riesgo_actual * COMMISSION_PER_TRADE * 2) # Nueva comisión
                    trade_pnl_net = -riesgo_actual - (riesgo_actual * COMMISSION_PER_TRADE * 2)
                    trades.append(('SHORT_PERDIDA', precio_entrada, stop_loss, fecha_entrada, fecha_salida, trade_pnl_net))
                    trade_executed = True
                    open_trade_details = None # Trade closed
                    break
                elif low_price <= take_profit: # Salida por Take Profit
                    profit = riesgo_actual * RR_RATIO
                    balance += profit
                    balance -= (riesgo_actual * COMMISSION_PER_TRADE * 2) # Nueva comisión
                    trade_pnl_net = profit - (riesgo_actual * COMMISSION_PER_TRADE * 2)
                    trades.append(('SHORT_GANANCIA', precio_entrada, take_profit, fecha_entrada, fecha_salida, trade_pnl_net))
                    trade_executed = True
                    open_trade_details = None # Trade closed
                    break
            # Si el bucle termina sin break (fin de datos), no se cierra la operación

        # Actualizar historial de balance solo si cambió (hubo trade) o al final
        # El código original lo añadía en cada iteración, ajustamos ligeramente
        # if trade_executed: # Opcional: añadir solo si hubo trade o siempre como antes
        balance_hist.append(balance) # Mantener como antes: añadir en cada paso i

    # Asegurar que el historial tenga la misma longitud que los pasos + inicial
    # if len(balance_hist) == len(df_processed): # El bucle va hasta len-1
    #      balance_hist.append(balance) # Añadir el último estado si es necesario

    # Log si un trade quedó abierto al final de los datos
    if open_trade_details is not None:
        logging.info(
            f"Note: A {open_trade_details['type']} position entered at {open_trade_details['entry_price']:.5f} "
            f"on {open_trade_details['entry_date']} (SL: {open_trade_details['sl']:.5f}, TP: {open_trade_details['tp']:.5f}) "
            f"was still conceptually open at the end of the backtest data and was not closed by this simulation."
        )

    logging.info(f"Backtesting finalizado. Balance final: {balance:.2f}")
    return balance, trades, balance_hist



# Función para confirmar la señal con indicadores técnicos (Optimizada)
def confirmar_senal_con_indicadores(
    current_rsi: float,
    current_macd_diff: float,
    current_ema_fast: float,
    current_ema_slow: float
) -> Tuple[bool, bool]:
    """
    Confirma señales LONG/SHORT basadas en los valores actuales de los indicadores.

    Args:
        current_rsi: Valor actual del RSI.
        current_macd_diff: Valor actual del MACD Diff.
        current_ema_fast: Valor actual de la EMA rápida.
        current_ema_slow: Valor actual de la EMA lenta.

    Returns:
        Una tupla (confirmar_long, confirmar_short) de booleanos.
    """
    # Definir condiciones de confirmación
    # No es necesario comprobar NaNs aquí, ya se hace antes de llamar
    confirmar_long: bool = current_rsi > 50 and current_macd_diff > 0 and current_ema_fast > current_ema_slow
    confirmar_short: bool = current_rsi < 50 and current_macd_diff < 0 and current_ema_fast < current_ema_slow

    return confirmar_long, confirmar_short


# Note: This function is for live trading position sizing and risk management based on available exchange balance.
# The backtesting P&L in realizar_backtest uses a simplified model based on a direct percentage of the simulated balance.
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
