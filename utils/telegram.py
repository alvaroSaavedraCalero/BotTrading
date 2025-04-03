
from config.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
import requests
import logging

# Función para enviar mensajes a Telegram
def enviar_mensaje_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje,
        "parse_mode": "HTML"    
    }
    try:
        response = requests.get(url=url, params=payload)
        if response.status_code == 200:
            logging.info("✅ Mensaje enviado a Telegram correctamente.")
        else:
            logging.info(f"⚠️ Error al enviar mensaje a Telegram. Código: {response.status_code}, Respuesta: {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Excepción al enviar mensaje a Telegram: {e}")

