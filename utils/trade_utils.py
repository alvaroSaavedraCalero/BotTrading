# utils/trade_utils.py

import logging
from typing import Optional, Tuple

def calcular_cantidad(
    riesgo: float,
    stop_loss: float,
    precio_entrada: float,
    saldo_usdt: float
) -> Optional[float]:
    """
    Calcula la cantidad de la orden basada en el riesgo porcentual sobre el saldo
    disponible en USDT y la distancia al stop loss.

    Args:
        riesgo: Riesgo porcentual a asumir sobre el saldo (ej. 0.01 para 1%).
        stop_loss: Precio del stop loss.
        precio_entrada: Precio de entrada de la operación.
        saldo_usdt: Saldo disponible en USDT

    Returns:
        La cantidad calculada redondeada, o None si ocurre un error o no hay saldo.
    """
    try:
        if saldo_usdt <= 0:
            logging.warning("⚠️ No hay saldo disponible en USDT.")
            return None

        riesgo_monetario: float = saldo_usdt * riesgo  # Riesgo total en USDT
        distancia_sl: float = abs(precio_entrada - stop_loss)

        if distancia_sl == 0:
            logging.warning("⚠️ Distancia al Stop Loss es cero. No se puede calcular cantidad.")
            return None

        cantidad: float = riesgo_monetario / distancia_sl
        # Redondeo (precisión puede depender del par, 6 es un ejemplo)
        return round(cantidad, 6)
    except Exception as e:
        logging.error(f"❌ Error al calcular cantidad: {e}", exc_info=True)
        return None

