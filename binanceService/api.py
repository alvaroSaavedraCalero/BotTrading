# binanceService/api.py
import logging
import pandas as pd
from binance.client import Client
from typing import List, Tuple, Optional, Dict, Any # Importar tipos necesarios

# Importar configuración y otras dependencias con tipos si es posible
from config.config import BINANCE_API_KEY, BINANCE_API_SECRET, INITIAL_CAPITAL, RISK_PERCENT
# Asumiendo que las funciones importadas también están tipadas o lo estarán
from backtestingService.engine import calcular_cantidad
from utils.telegram import enviar_mensaje_telegram

# Verificar que las claves API existen antes de crear el cliente
if BINANCE_API_KEY is None or BINANCE_API_SECRET is None:
    logging.error("❌ Las claves API de Binance no están definidas en la configuración.")
    # Podrías lanzar un error o manejarlo de otra forma
    raise ValueError("Faltan las claves API de Binance.")
else:
    # Tipar la instancia del cliente
    client: Optional[Client] = None
    if BINANCE_API_KEY == "dummy_api_key" and BINANCE_API_SECRET == "dummy_api_secret":
        logging.warning("Using dummy API keys. Binance client will not be fully initialized for live operations.")
        # We can't fully mock the client without more extensive changes,
        # but we can prevent it from making immediate calls or assign a mock object if needed.
        # For now, let's assume parts of the code that use `client` directly might fail
        # if they expect a live connection, e.g., live trading.
        # The `obtener_datos_binance` will be mocked to allow backtesting.
        pass # client remains None or could be a mock object
    else:
        try:
            client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        except Exception as e:
            logging.error(f"Failed to initialize Binance Client with provided keys: {e}")
            client = None # Ensure client is None if initialization fails


# Funcion para obtener los datos de binance
def obtener_datos_binance(simbolo: str = 'BTCUSDT', intervalo: str = '15m', periodo: str = '1 month ago UTC') -> pd.DataFrame:
    """
    Obtiene datos históricos OHLCV de Binance para un símbolo, intervalo y periodo dados.
    Returns mock data if dummy API keys are in use.

    Args:
        simbolo: Símbolo del par (ej. 'BTCUSDT').
        intervalo: Intervalo de las velas (ej. '15m', '1h').
        periodo: Periodo histórico a obtener (ej. '1 month ago UTC').

    Returns:
        Un DataFrame de pandas con columnas 'open', 'high', 'low', 'close', 'temporalidad',
        indexado por 'open_time' (Timestamp). Retorna un DataFrame vacío si hay error.
    """
    logging.info(f"Obteniendo datos históricos de Binance para {simbolo} ({intervalo}, {periodo})...")

    if BINANCE_API_KEY == "dummy_api_key":
        logging.warning("Using DUMMY API Key - Returning MOCK Binance data for testing.")
        # Create a mock DataFrame
        mock_data = {
            'open_time': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:15:00', '2023-01-01 00:30:00', '2023-01-01 00:45:00'] * 1000), # Approx 2.7 days of 15m data
            'open': [16500.0, 16501.0, 16502.0, 16500.0] * 1000,
            'high': [16510.0, 16511.0, 16512.0, 16505.0] * 1000,
            'low': [16490.0, 16491.0, 16492.0, 16495.0] * 1000,
            'close': [16501.0, 16502.0, 16500.0, 16503.0] * 1000,
            # 'volume': [10.0, 11.0, 12.0, 13.0] * 1000, # Not strictly needed by current features
        }
        df_mock = pd.DataFrame(mock_data)
        df_mock.set_index('open_time', inplace=True)
        temporalidad_dict: Dict[str, int] = {'1m': 1, '5m': 2, '15m': 3, '30m': 4, '1h': 5, '4h': 6, '1d': 7, '1w': 8}
        df_mock['temporalidad'] = temporalidad_dict.get(intervalo, 0)
        # Ensure enough data for typical indicator periods (e.g., EMA 21)
        # Need at least 21 periods for EMA21. Let's make it much larger to be safe for various features.
        num_entries = 200 # Ensures enough data for most indicators
        data_points = {
            'open_time': pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=num_entries, freq=intervalo),
            'open': [16500.0 + i*0.1 for i in range(num_entries)],
            'high': [16510.0 + i*0.1 for i in range(num_entries)],
            'low': [16490.0 - i*0.1 for i in range(num_entries)],
            'close': [16505.0 + (i%5 - 2.5) for i in range(num_entries)], # Add some variation
        }
        df_mock_large = pd.DataFrame(data_points)
        df_mock_large.set_index('open_time', inplace=True)
        df_mock_large['temporalidad'] = temporalidad_dict.get(intervalo,0)
        logging.info(f"Mock data generated with {len(df_mock_large)} rows.")
        return df_mock_large.copy() # Return a copy

    try:
        if client is None:
            logging.error("Binance client is not initialized. Cannot fetch real data.")
            return pd.DataFrame()

        # get_historical_klines devuelve una lista de listas
        klines: List[List[Any]] = client.get_historical_klines(simbolo, intervalo, periodo)

        if not klines:
            logging.warning(f"No se recibieron datos de klines para {simbolo}, {intervalo}, {periodo}.")
            return pd.DataFrame() # Retornar DataFrame vacío

        # Definir columnas según la documentación de la API de Binance
        cols: List[str] = ['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore']
        df: pd.DataFrame = pd.DataFrame(klines, columns=cols)

        # Seleccionar y convertir columnas a numéricas (float)
        ohlc_cols: List[str] = ['open', 'high', 'low', 'close']
        df = df[['open_time'] + ohlc_cols] # Incluir open_time para conversión posterior
        # Convertir a float, manejar errores por si alguna columna no es convertible
        for col in ohlc_cols:
             df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convertir el tiempo de apertura a datetime y establecerlo como índice
        # Usar errors='coerce' por si hay timestamps inválidos
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
        df.dropna(subset=['open_time'], inplace=True) # Eliminar filas donde el timestamp falló
        df.set_index('open_time', inplace=True)

        # Añadir columna de temporalidad numérica
        temporalidad_dict: Dict[str, int] = {'1m': 1, '5m': 2, '15m': 3, '30m': 4, '1h': 5, '4h': 6, '1d': 7, '1w': 8}
        df['temporalidad'] = temporalidad_dict.get(intervalo, 0) # 0 si no está en el diccionario

        # Eliminar filas donde la conversión a float de OHLC pudo fallar
        df.dropna(subset=ohlc_cols, inplace=True)

        logging.info(f"Datos obtenidos correctamente para {simbolo}. {len(df)} filas.")
        return df

    except Exception as e:
         logging.error(f"❌ Error al obtener datos históricos de Binance: {e}", exc_info=True)
         return pd.DataFrame() # Retornar DataFrame vacío en caso de error


# Obtener datos actuales del mercado en tiempo real (última vela)
def obtener_datos_tiempo_real(simbolo: str, intervalo: str) -> Optional[pd.DataFrame]:
    """
    Obtiene la última vela (o vela actual incompleta) para un símbolo e intervalo.

    Args:
        simbolo: Símbolo del par.
        intervalo: Intervalo de la vela.

    Returns:
        Un DataFrame de pandas con una fila (OHLC) o None si hay error.
    """
    try:
        klines: List[List[Any]] = client.get_klines(symbol=simbolo, interval=intervalo, limit=1) # type: ignore
        if not klines:
            logging.warning(f"No se recibieron datos de klines en tiempo real para {simbolo}, {intervalo}.")
            return None

        cols: List[str] = ['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore']
        df: pd.DataFrame = pd.DataFrame(klines, columns=cols)
        ohlc_cols: List[str] = ['open', 'high', 'low', 'close']
        df = df[ohlc_cols]
        for col in ohlc_cols:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        if df.empty:
             logging.warning(f"DataFrame vacío después de procesar datos en tiempo real para {simbolo}.")
             return None

        return df

    except Exception as e:
        logging.error(f"❌ Error al obtener datos en tiempo real de Binance: {e}", exc_info=True)
        return None


# Función para consultar saldo de un activo específico
def consultar_saldo_especifico(asset: str = "USDT") -> Tuple[Optional[float], Optional[float]]:
    """
    Consulta el saldo libre y bloqueado para un activo específico en Binance.

    Args:
        asset: El símbolo del activo (ej. "USDT", "BTC").

    Returns:
        Una tupla (saldo_disponible, saldo_bloqueado). Retorna (None, None) si hay error.
    """
    try:
        # get_asset_balance devuelve un diccionario o None
        saldo: Optional[Dict[str, str]] = client.get_asset_balance(asset=asset)

        saldo_disponible: Optional[float] = None
        saldo_bloqueado: Optional[float] = None

        if saldo is not None:
            # Los valores vienen como string, convertir a float
            saldo_disponible = float(saldo['free'])
            saldo_bloqueado = float(saldo['locked'])
            logging.info(f"✅ Saldo de {asset}: Disponible={saldo_disponible:.4f}, Bloqueado={saldo_bloqueado:.4f}")
        else:
            # Si el activo no existe en la cuenta, el saldo es 0
            saldo_disponible = 0.0
            saldo_bloqueado = 0.0
            logging.info(f"ℹ️ No se encontró saldo para el activo {asset} (o es cero).")

        return saldo_disponible, saldo_bloqueado

    except Exception as e:
        logging.error(f"❌ Error al consultar saldo de {asset}: {e}", exc_info=True)
        return None, None


# Función para consultar saldo de todos los activos con balance > 0
def consultar_saldo_completo() -> Optional[List[Tuple[str, float, float]]]:
    """
    Consulta los saldos de todos los activos en la cuenta que tengan balance > 0.

    Returns:
        Una lista de tuplas (asset, free_balance, locked_balance) o None si hay error.
    """
    try:
        # get_account devuelve un diccionario complejo, ['balances'] es una lista de dicts
        balances: List[Dict[str, str]] = client.get_account()['balances']

        activos_con_saldo: List[Tuple[str, float, float]] = []
        logging.info("\n📊 **SALDO DISPONIBLE EN BINANCE** 📊")
        for activo in balances:
            free: float = float(activo['free'])
            locked: float = float(activo['locked'])
            if free > 0 or locked > 0:
                asset_name: str = activo['asset']
                activos_con_saldo.append((asset_name, free, locked))
                logging.info(f"🔹 {asset_name}: {free:.4f} disponible | {locked:.4f} bloqueado")

        return activos_con_saldo

    except Exception as e:
        logging.error(f"❌ Error al consultar saldo completo: {e}", exc_info=True)
        return None


# Función para ejecutar órdenes en Binance (Compra / Venta con Stop Loss y Take Profit OCO)
def ejecutar_orden_binance(simbolo: str, tipo: str, precio_entrada: float, stop_loss: float, take_profit: float) -> None:
    """
    Ejecuta una orden de mercado (BUY o SELL) y luego intenta colocar una orden OCO
    para establecer el Stop Loss y Take Profit.

    Args:
        simbolo: Símbolo del par (ej. 'BTCUSDT').
        tipo: Tipo de orden ('LONG' o 'SHORT').
        precio_entrada: Precio estimado de entrada.
        stop_loss: Precio de Stop Loss.
        take_profit: Precio de Take Profit.

    Returns:
        None
    """
    try:
        saldo_usdt: Optional[float]
        saldo_usdt, _ = consultar_saldo_especifico("USDT")

        # Validar saldo antes de proceder
        if saldo_usdt is None:
             logging.error("❌ No se pudo obtener el saldo USDT. No se ejecutará la orden.")
             return
        if saldo_usdt < (INITIAL_CAPITAL * RISK_PERCENT): # Comprobar si hay suficiente para el riesgo mínimo
            logging.warning(f"⚠️ Saldo USDT ({saldo_usdt:.2f}) insuficiente para cubrir el riesgo inicial mínimo. No se abre operación.")
            return

        # Calcular cantidad usando la función del engine (que ya retorna Optional[float])
        cantidad: Optional[float] = calcular_cantidad(RISK_PERCENT, stop_loss, precio_entrada)

        if cantidad is None or cantidad <= 0:
            logging.error(f"❌ Cantidad calculada inválida ({cantidad}). No se ejecutará la orden.")
            return

        # Formatear cantidad según reglas del símbolo (esto requiere otra llamada API o info precargada)
        # Por ahora, asumimos un redondeo genérico, pero esto puede fallar.
        # Idealmente, obtener stepSize de exchangeInfo y redondear.
        # TODO: Implement proper quantity formatting using exchangeInfo.
        # The current rounding is a placeholder and may lead to API errors for symbols
        # with specific lot size/step size requirements. Fetch exchangeInfo for the symbol,
        # parse the 'LOT_SIZE' filter (specifically 'stepSize'), and use that to correctly
        # quantize the 'cantidad' before placing the order.
        cantidad_formateada: float = round(cantidad, 5) # Ejemplo de redondeo, AJUSTAR SEGÚN PAR
        logging.warning(f"Using generically rounded quantity {cantidad_formateada} for {simbolo}. This may not meet the symbol's precise stepSize requirement and could cause order failure in live trading. Implement exchangeInfo-based formatting.")
        if cantidad_formateada <= 0:
             logging.error(f"❌ Cantidad formateada es cero o negativa ({cantidad_formateada}). No se ejecutará orden.")
             return

        # Detailed logging before placing the order
        distancia_sl = abs(precio_entrada - stop_loss)
        # Ensure saldo_usdt is not None here for fallback, though it should be checked earlier.
        # If cantidad_formateada is valid and distancia_sl is 0, it implies an issue with SL price.
        # The fallback saldo_usdt * RISK_PERCENT is a rough estimate of intended risk if SL is at entry.
        riesgo_monetario_calculado = cantidad_formateada * distancia_sl if distancia_sl > 0 else (saldo_usdt * RISK_PERCENT if saldo_usdt else 0)

        logging.info(f"Preparing Live Order: Symbol={simbolo}, Type={tipo}, Quantity={cantidad_formateada:.8f}, EntryPriceEst={precio_entrada:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}, MonetaryRiskEst={riesgo_monetario_calculado:.2f} USDT")
        logging.info(f"Intentando ejecutar orden {tipo} para {simbolo} con cantidad {cantidad_formateada:.5f}") # Original log

        # Ejecutar orden de mercado principal
        orden_principal: Optional[Dict[str, Any]] = None
        side: str = "" # Para OCO order
        if tipo == "LONG":
            side = Client.SIDE_SELL # OCO para cerrar un LONG es SELL
            orden_principal = client.order_market_buy(symbol=simbolo, quantity=cantidad_formateada)
        elif tipo == "SHORT":
            side = Client.SIDE_BUY # OCO para cerrar un SHORT es BUY
            orden_principal = client.order_market_sell(symbol=simbolo, quantity=cantidad_formateada)
        else:
             logging.error(f"Tipo de orden '{tipo}' no reconocido.")
             return

        if orden_principal:
            # Obtener la cantidad ejecutada real de la orden principal si es posible/necesario
            # La API puede devolver 'executedQty'
            cantidad_ejecutada_str: str = orden_principal.get('executedQty', str(cantidad_formateada))
            try:
                cantidad_ejecutada: float = float(cantidad_ejecutada_str)
                logging.info(f"✅ Orden Principal {tipo} Ejecutada: {orden_principal}")
            except ValueError:
                logging.error(f"No se pudo convertir executedQty '{cantidad_ejecutada_str}' a float. Usando cantidad formateada para OCO.")
                cantidad_ejecutada = cantidad_formateada


            # Configurar Stop Loss y Take Profit con OCO Order
            # Asegurarse que los precios SL/TP se pasan como string
            # stopLimitPrice debe ser ligeramente diferente a stopPrice para órdenes tipo STOP_LOSS_LIMIT
            # Para OCO, si stopPrice se alcanza, se coloca una orden LIMIT a stopLimitPrice
            sl_limit_price: float
            if tipo == "LONG": # SL es venta, límite debe ser <= stopPrice
                 sl_limit_price = stop_loss * 0.998 # Un poco por debajo para asegurar ejecución límite
            else: # SL es compra, límite debe ser >= stopPrice
                 sl_limit_price = stop_loss * 1.002 # Un poco por encima

            logging.info(f"Intentando colocar orden OCO {side} para cerrar posición...")
            logging.info(f"Params OCO: symbol={simbolo}, quantity={cantidad_ejecutada}, price(TP)={str(take_profit)}, stopPrice(SL)={str(stop_loss)}, stopLimitPrice={str(sl_limit_price)}")

            orden_oco: Dict[str, Any] = client.create_oco_order(
                symbol=simbolo,
                side=side, # 'SELL' para cerrar LONG, 'BUY' para cerrar SHORT
                quantity=cantidad_ejecutada,
                price=f"{take_profit:.8f}", # Precio Límite (Take Profit) - formatear a string con precisión adecuada
                stopPrice=f"{stop_loss:.8f}", # Precio Stop (Stop Loss) - formatear
                stopLimitPrice=f"{sl_limit_price:.8f}", # Precio Límite del Stop - formatear
                stopLimitTimeInForce=Client.TIME_IN_FORCE_GTC # Good Til Cancelled
            )
            logging.info(f"✅ Orden OCO (SL/TP) colocada: {orden_oco}")

            mensaje: str = (
                f"📌 🚀 ORDEN EJECUTADA {tipo} en Binance\n"
                f"📊 Símbolo: {simbolo}\n"
                f"🔹 Entrada Aprox: {precio_entrada:.5f}\n" # Mostrar más precisión
                f"🛡️ Stop Loss: {stop_loss:.5f}\n"
                f"🎯 Take Profit: {take_profit:.5f}\n"
                f"⚖️ Cantidad: {cantidad_ejecutada:.5f}"
            )
            enviar_mensaje_telegram(mensaje)
        else:
             logging.error(f"❌ La orden principal {tipo} no parece haber sido ejecutada.")


    except Exception as e:
        # Capturar específicamente errores de Binance API si es posible
        # from binance.exceptions import BinanceAPIException, BinanceOrderException
        # if isinstance(e, BinanceAPIException):
        #     logging.error(f"❌ Error API Binance: {e.status_code} - {e.message}")
        # elif isinstance(e, BinanceOrderException):
        #      logging.error(f"❌ Error Orden Binance: {e.code} - {e.message}")
        # else:
        logging.error(f"⚠️ Error general al ejecutar orden en Binance: {e}", exc_info=True)
        # Intentar enviar error a Telegram
        try:
             enviar_mensaje_telegram(f"❌ Error crítico al intentar ejecutar orden {tipo} en {simbolo} de Binance: {e}")
        except Exception as te:
             logging.error(f"Fallo al enviar mensaje de error a Telegram: {te}")