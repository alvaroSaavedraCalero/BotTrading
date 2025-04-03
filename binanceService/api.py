import logging
import pandas as pd
from binance.client import Client
from config.config import BINANCE_API_KEY, BINANCE_API_SECRET, INITIAL_CAPITAL, RISK_PERCENT
from backtestingService.engine import calcular_cantidad
from utils.telegram import enviar_mensaje_telegram

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)


# Funcion para obtener los datos de binance
def obtener_datos_binance(simbolo='BTCUSDT', intervalo='15m', periodo='1 month ago UTC'):
    
    logging.info("Obteniendo datos históricos de Binance...")
    klines = client.get_historical_klines(simbolo, intervalo, periodo)
    df = pd.DataFrame(klines, columns=['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore'])
    
    # Convertir columnas a numéricas
    df = df[['open_time', 'open', 'high', 'low', 'close']].astype(float)
    
    # Convertir el tiempo de apertura a datetime y establecerlo como índice
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    
    temporalidad_dict = {'1m': 1, '5m': 2, '15m': 3, '30m': 4, '1h': 5, '4h': 6, '1d': 7, '1w': 8}
    df['temporalidad'] = temporalidad_dict.get(intervalo, 0)  # 0 si no está en el diccionario
    
    logging.info("Datos obtenidos correctamente")
    return df


# Obtener datos actuales del mercado en tiempo real
def obtener_datos_tiempo_real(simbolo, intervalo):
    klines = client.get_klines(symbol=simbolo, interval=intervalo, limit=1)
    df = pd.DataFrame(klines, columns=['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore'])
    df = df[['open', 'high', 'low', 'close']].astype(float)
    return df




# Función para consultar saldo de un activo específico
def consultar_saldo_especifico(asset="USDT"):
    try:
        saldo = client.get_asset_balance(asset=asset)
        if saldo is not None:
            saldo_disponible = float(saldo['free'])
            saldo_bloqueado = float(saldo['locked'])
        else:
            saldo_disponible = 0.0
            saldo_bloqueado = 0.0
        
        logging.info(f"✅ Saldo de {asset} disponible: {saldo_disponible:.2f}, Bloqueado: {saldo_bloqueado:.2f}") 
        return saldo_disponible, saldo_bloqueado
    
    except Exception as e:
        logging.error(f"❌ Error al consultar saldo de {asset}: {e}")
        return None, None




# Función para consultar saldo de todos los activos
def consultar_saldo_completo():
    try:
        balances = client.get_account()['balances']
        activos_con_saldo = [
            (activo['asset'], float(activo['free']), float(activo['locked']))
            for activo in balances if float(activo['free']) > 0 or float(activo['locked']) > 0
        ]
        logging.info("\n📊 **SALDO DISPONIBLE EN BINANCE** 📊")
        for asset, free, locked in activos_con_saldo:
            logging.info(f"🔹 {asset}: {free} disponible | {locked} bloqueado en órdenes")
        return activos_con_saldo
    except Exception as e:
        logging.error(f"❌ Error al consultar saldo de todos los activos: {e}")
        return None



# Función para ejecutar órdenes en Binance (Compra / Venta con Stop Loss y Take Profit)
def ejecutar_orden_binance(simbolo, tipo, precio_entrada, stop_loss, take_profit):
    try:
        saldo_usdt, _ = consultar_saldo_especifico("USDT")
        if saldo_usdt is None or saldo_usdt < (INITIAL_CAPITAL * RISK_PERCENT):
            logging.warning("⚠️ Saldo insuficiente para abrir la operación.")
            return
        cantidad = calcular_cantidad(RISK_PERCENT, stop_loss, precio_entrada)
        if cantidad is None or cantidad <= 0:
            logging.error("❌ Cantidad calculada inválida. No se ejecutará la orden.")
            return
        orden = None
        if tipo == "LONG":
            orden = client.order_market_buy(symbol=simbolo, quantity=cantidad)
        elif tipo == "SHORT":
            orden = client.order_market_sell(symbol=simbolo, quantity=cantidad)
        if orden:
            logging.info(f"✅ {tipo} Ejecutado: {orden}")
            # Configurar Stop Loss y Take Profit con OCO Order
            client.order_oco_sell(
                symbol=simbolo,
                quantity=cantidad,
                price=str(take_profit),
                stopPrice=str(stop_loss),
                stopLimitPrice=str(stop_loss * 0.995),  # Reducido para evitar rechazo
                stopLimitTimeInForce="GTC"
            )
            mensaje = (
                f"📌 🚀 ORDEN EJECUTADA {tipo} en Binance\n"
                f"🔹 Entrada: {precio_entrada:.2f}\n"
                f"🔹 Stop Loss: {stop_loss:.2f}\n"
                f"🔹 Take Profit: {take_profit:.2f}\n"
                f"🔹 Cantidad: {cantidad} {simbolo}"
            )
            enviar_mensaje_telegram(mensaje)
    except Exception as e:
        logging.error(f"⚠️ Error al ejecutar orden en Binance: {e}")
        enviar_mensaje_telegram(f"❌ Error al ejecutar orden en Binance: {e}")

