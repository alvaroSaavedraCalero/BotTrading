
import logging
from config.config import RISK_PERCENT, RR_RATIO
from binanceService.api import ejecutar_orden_binance
from backtestingService.engine import confirmar_senal_con_indicadores

# Generar señal claramente estructurada
def generar_senal_tiempo_real(model, df, simbolo, modo_simulacion=True):
    features = df[['open', 'high', 'low', 'close']]
    prediccion = model.predict(features)[0]
    precio_actual = df['close'].iloc[-1]

    # Confirmación con análisis técnico
    confirmar_long, confirmar_short = confirmar_senal_con_indicadores(df)

    if prediccion == 2 and confirmar_long:
        stop_loss = precio_actual * (1 - RISK_PERCENT)
        take_profit = precio_actual * (1 + RISK_PERCENT * RR_RATIO)
        if modo_simulacion:
            logging.info(f"💡 Confirmada: LONG {simbolo} | Entrada: {precio_actual:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
        else:
            ejecutar_orden_binance(simbolo, "LONG", precio_actual, stop_loss, take_profit)
        return {"Tipo": "LONG 🚀", "Precio Entrada": precio_actual, "Stop Loss": round(stop_loss, 2), "Take Profit": round(take_profit, 2)}

    elif prediccion == 0 and confirmar_short:
        stop_loss = precio_actual * (1 + RISK_PERCENT)
        take_profit = precio_actual * (1 - RISK_PERCENT * RR_RATIO)
        if modo_simulacion:
            logging.info(f"💡 Confirmada: SHORT {simbolo} | Entrada: {precio_actual:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
        else:
            ejecutar_orden_binance(simbolo, "SHORT", precio_actual, stop_loss, take_profit)
        return {"Tipo": "SHORT 📉", "Precio Entrada": precio_actual, "Stop Loss": round(stop_loss, 2), "Take Profit": round(take_profit, 2)}

    logging.info("🔍 Señal NO confirmada por indicadores técnicos.")
    return {"Tipo": "NEUTRAL ⚪️", "Precio Entrada": None, "Stop Loss": None, "Take Profit": None}
