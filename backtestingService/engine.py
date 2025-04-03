
from config.config import INITIAL_CAPITAL, RISK_PERCENT, RR_RATIO, COMMISSION_PER_TRADE, SIMULATED_SPREAD
import logging
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator

import binanceService.api as binance_api


# Backtesting y simulación de resultados financieros
def realizar_backtest(df, model):
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, EMAIndicator
    from ta.volatility import AverageTrueRange

    logging.info("Realizando backtesting y simulación de resultados mejorada...")

    df['returns'] = df['close'].pct_change()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = MACD(df['close']).macd_diff()
    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    df = df.dropna()

    balance = INITIAL_CAPITAL
    balance_hist = [balance]
    trades = []

    for i in range(len(df) - 1):
        riesgo_actual = balance * RISK_PERCENT
        features = df[['open', 'high', 'low', 'close', 'temporalidad', 'returns', 'rsi', 'macd', 'ema9', 'ema21', 'atr']].iloc[i:i+1].fillna(0)
        prediccion = model.predict(features)[0]
        fecha_entrada = df.index[i]

        sub_df = df.iloc[:i+1]
        confirmar_long, confirmar_short = confirmar_senal_con_indicadores(sub_df)

        if prediccion == 2 and confirmar_long:
            precio_entrada = df['close'].iloc[i] * (1 + SIMULATED_SPREAD)
            stop_loss = precio_entrada * (1 - RISK_PERCENT)
            take_profit = precio_entrada * (1 + RISK_PERCENT * RR_RATIO)

            for j in range(i + 1, len(df)):
                fecha_salida = df.index[j]
                if df['low'].iloc[j] <= stop_loss:
                    balance -= riesgo_actual * (1 + COMMISSION_PER_TRADE * 2)
                    trades.append(('LONG_PERDIDA', precio_entrada, stop_loss, fecha_entrada, fecha_salida))
                    break
                elif df['high'].iloc[j] >= take_profit:
                    balance += riesgo_actual * RR_RATIO * (1 - COMMISSION_PER_TRADE * 2)
                    trades.append(('LONG_GANANCIA', precio_entrada, take_profit, fecha_entrada, fecha_salida))
                    break

        elif prediccion == 0 and confirmar_short:
            precio_entrada = df['close'].iloc[i] * (1 - SIMULATED_SPREAD)
            stop_loss = precio_entrada * (1 + RISK_PERCENT)
            take_profit = precio_entrada * (1 - RISK_PERCENT * RR_RATIO)

            for j in range(i + 1, len(df)):
                fecha_salida = df.index[j]
                if df['high'].iloc[j] >= stop_loss:
                    balance -= riesgo_actual * (1 + COMMISSION_PER_TRADE * 2)
                    trades.append(('SHORT_PERDIDA', precio_entrada, stop_loss, fecha_entrada, fecha_salida))
                    break
                elif df['low'].iloc[j] <= take_profit:
                    balance += riesgo_actual * RR_RATIO * (1 - COMMISSION_PER_TRADE * 2)
                    trades.append(('SHORT_GANANCIA', precio_entrada, take_profit, fecha_entrada, fecha_salida))
                    break

        balance_hist.append(int(balance))

    return balance, trades, balance_hist



# Función para confirmar la señal con indicadores técnicos
def confirmar_senal_con_indicadores(df):
    df = df.copy()

    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['macd_diff'] = macd.macd_diff()
    df['ema_fast'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema_slow'] = EMAIndicator(df['close'], window=21).ema_indicator()

    rsi = df['rsi'].iloc[-1]
    macd_val = df['macd_diff'].iloc[-1]
    ema_fast = df['ema_fast'].iloc[-1]
    ema_slow = df['ema_slow'].iloc[-1]

    confirmar_long = rsi > 50 and macd_val > 0 and ema_fast > ema_slow
    confirmar_short = rsi < 50 and macd_val < 0 and ema_fast < ema_slow

    return confirmar_long, confirmar_short


# Función para calcular tamaño de la orden basado en el riesgo y balance actual
def calcular_cantidad(riesgo, stop_loss, precio_entrada):
    try:
        saldo_usdt, _ = binance_api.consultar_saldo_especifico("USDT")  # Balance en USDT
        if saldo_usdt is None or saldo_usdt <= 0:
            logging.warning("⚠️ No hay suficiente saldo en USDT.")
            return None
        riesgo_total = saldo_usdt * riesgo  # Riesgo basado en el saldo USDT
        cantidad = riesgo_total / abs(precio_entrada - stop_loss)
        return round(cantidad, 6)  # Redondeo según Binance
    except Exception as e:
        logging.error(f"❌ Error al calcular cantidad: {e}")
        return None
