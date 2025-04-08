# utils/telegram.py

import logging
import requests
# Importar excepciones específicas de requests para el bloque try-except
from requests.exceptions import RequestException
from typing import Dict, Optional # Importar tipos necesarios

# Importar configuración (variables son Optional[str])
from config.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

# Función para enviar mensajes a Telegram
def enviar_mensaje_telegram(mensaje: str) -> None:
    """
    Envía un mensaje de texto al chat de Telegram especificado en la configuración.

    Args:
        mensaje: El contenido del mensaje a enviar (string).

    Returns:
        None
    """
    # --- Comprobación de Configuración ---
    # Es crucial verificar que el token y el chat_id no sean None antes de usarlos
    if not TELEGRAM_TOKEN:
        logging.error("❌ No se puede enviar mensaje a Telegram: TELEGRAM_TOKEN no está configurado.")
        return # Salir de la función si falta el token
    if not TELEGRAM_CHAT_ID:
        logging.error("❌ No se puede enviar mensaje a Telegram: TELEGRAM_CHAT_ID no está configurado.")
        return # Salir de la función si falta el chat ID

    # --- Construcción de la Petición ---
    # Ahora sabemos que TELEGRAM_TOKEN no es None
    url: str = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    # El payload espera strings para todos los valores
    payload: Dict[str, str] = {
        "chat_id": TELEGRAM_CHAT_ID, # Ya sabemos que no es None
        "text": mensaje,
        "parse_mode": "HTML" # Modo de parseo para el texto
    }

    # --- Envío y Manejo de Respuesta ---
    try:
        # response es de tipo requests.Response
        response: requests.Response = requests.get(url=url, params=payload, timeout=10) # Añadir timeout

        # Comprobar código de estado
        if response.status_code == 200:
            logging.info("✅ Mensaje enviado a Telegram correctamente.")
        else:
            # Loggear más detalles en caso de error
            logging.warning(
                f"⚠️ Error al enviar mensaje a Telegram. Código: {response.status_code}. "
                f"Respuesta: {response.text}" # response.text es string
            )

    # Capturar excepciones de la librería requests
    except RequestException as e:
        logging.error(f"❌ Excepción al enviar mensaje a Telegram: {e}", exc_info=True)
    except Exception as e_general:
         # Capturar cualquier otra excepción inesperada
         logging.error(f"❌ Error inesperado en enviar_mensaje_telegram: {e_general}", exc_info=True)