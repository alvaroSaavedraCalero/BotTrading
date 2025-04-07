# models/builder.py

import logging
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os
from datetime import datetime
from utils.enumerados import Modelo

from config.config import RANDOM_STATE, N_ESTIMATORS
from binanceService.api import obtener_datos_binance
from models.trainer import entrenar_modelo


# Funcion para obtener un modelo de 0 sin entrenamiento
def generar_modelo_nuevo(modelo=Modelo.RANDOM_FOREST):
    if modelo == Modelo.GRADIENT_BOOSTING:
        logging.info("Generando nuevo modelo GradientBoostingClassifier...")
        modelo = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    elif modelo == Modelo.RANDOM_FOREST:
        logging.info("Generando nuevo modelo RandomForestClassifier...")
        modelo = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    elif modelo == Modelo.XGB:
        logging.info("Generando nuevo modelo XGBClassifier...")
        modelo = XGBClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, objective='multi:softprob', num_class=3)
    else:
        raise ValueError(f"Tipo de modelo '{modelo.name}' no soportado.")
    
    return modelo


# Funcion para guardar un modelo entrenado
def guardar_modelo(model, nombre_modelo, simbolo):
    ruta = os.path.expanduser(rf"C:\Users\Álvaro\OneDrive\Escritorio\InfoRecursosBots\ModelosEntrenados")  # Carpeta accesible en cualquier sistema
    os.makedirs(ruta, exist_ok=True)  # Crea la carpeta si no existe

    fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nombre_archivo = f"modelo_entrenado_{nombre_modelo}_{simbolo}_{fecha_hora}.joblib"
    ruta_final = os.path.join(ruta, nombre_archivo)

    joblib.dump(model, ruta_final)
    logging.info(f"✅ Modelo guardado en '{ruta_final}'")
    
    
# Funcion para cargar un modelo entrenado
def cargar_modelo(ruta):
    if os.path.exists(ruta):
        logging.info(f"Cargando modelo desde '{ruta}'...")
        modelo = joblib.load(ruta)
        return modelo
    else:
        raise FileNotFoundError(f"No existe el modelo en '{ruta}'.")
    
    
# Funcion para entenar un modelo creado, o crear uno y entrenarlo
def crear_entrenar_modelo(ruta_modelo, simbolo, temporalidad, periodo, modelo=Modelo.RANDOM_FOREST):
    if ruta_modelo is None:
        model = generar_modelo_nuevo(modelo=modelo)
    else:
        try:
            logging.info("Cargando modelo...")
            model = cargar_modelo(ruta_modelo)
        except FileNotFoundError:
            logging.info("Modelo no encontrado, generando nuevo modelo...")
            model = generar_modelo_nuevo(modelo=modelo)
    
        
    logging.info("Obteniendo datos...")
    df = obtener_datos_binance(simbolo=simbolo, intervalo=temporalidad, periodo=periodo)
    
    logging.info("Entrenando modelo...")
    model = entrenar_modelo(model, df)
    
    logging.info("Guardando modelo...")
    guardar_modelo(model, modelo.name, simbolo)
    logging.info("Modelo gurdado correctamente")
    
    return model