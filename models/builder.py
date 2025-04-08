# models/builder.py

import logging
import os
from datetime import datetime
from typing import Union, Optional, Any # Importar tipos necesarios
import pandas as pd # Necesario para anotar el tipo de df
import joblib

# Importar clases específicas de modelos para anotaciones
from sklearn.base import BaseEstimator # O usar clases específicas si prefieres
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

# Importar Enum y config
from utils.enumerados import Modelo # Asumiendo que Modelo es un Enum
from utils.enumerados import TargetMethod # Asumiendo que Target es un Enum
from config.config import RANDOM_STATE, N_ESTIMATORS, MODEL_SAVE_DIR # Usar dir de config si es apropiado

# Importar funciones de otros módulos (asumiendo que están/estarán tipadas)
from binanceService.api import obtener_datos_binance
from models.trainer import entrenar_modelo

# Definir un alias de tipo para los posibles modelos generados/entrenados
# Podrías usar BaseEstimator si todos heredan de él, o ser más específico
ModelType = Union[GradientBoostingClassifier, RandomForestClassifier, XGBClassifier, BaseEstimator, Any]


# Funcion para obtener un modelo de 0 sin entrenamiento
def generar_modelo_nuevo(modelo_enum: Modelo = Modelo.RANDOM_FOREST) -> ModelType:
    """
    Genera una nueva instancia de un modelo de clasificación sin entrenar.

    Args:
        modelo_enum: El tipo de modelo a generar, desde el Enum Modelo.

    Returns:
        Una instancia del modelo de scikit-learn o XGBoost especificado.
        Retorna Any o un tipo base si no se puede determinar específicamente.

    Raises:
        ValueError: Si el tipo de modelo no es soportado.
    """
    model_instance: ModelType # Variable para almacenar la instancia
    if modelo_enum == Modelo.GRADIENT_BOOSTING:
        logging.info("Generando nuevo modelo GradientBoostingClassifier...")
        model_instance = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    elif modelo_enum == Modelo.RANDOM_FOREST:
        logging.info("Generando nuevo modelo RandomForestClassifier...")
        model_instance = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1) # Añadir n_jobs
    elif modelo_enum == Modelo.XGB:
        logging.info("Generando nuevo modelo XGBClassifier...")
        # Asegurarse que los parámetros coincidan con los usados en trainer si es relevante
        model_instance = XGBClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                       objective='multi:softprob', num_class=3, n_jobs=-1) # Añadir n_jobs
    else:
        # Usar modelo_enum.name si existe en el Enum para el mensaje de error
        raise ValueError(f"Tipo de modelo '{getattr(modelo_enum, 'name', modelo_enum)}' no soportado.")

    return model_instance


# Funcion para guardar un modelo entrenado
def guardar_modelo(model: Any, nombre_base_modelo: str, simbolo: str) -> None:
    """
    Guarda un modelo entrenado en disco usando joblib.

    Args:
        model: El objeto del modelo entrenado a guardar. Se usa Any por flexibilidad.
        nombre_base_modelo: Un nombre base para el archivo (ej. 'RANDOM_FOREST').
        simbolo: El símbolo para el cual se entrenó el modelo (ej. 'BTCUSDT').

    Returns:
        None
    """
    # Usar la ruta definida en config.py si es apropiado y existe
    # ruta: str = MODEL_SAVE_DIR
    # O mantener la ruta definida aquí si es específica para esta función
    ruta: str = os.path.expanduser(rf"C:\Users\Álvaro\OneDrive\Escritorio\InfoRecursosBots\ModelosEntrenados")
    try:
        os.makedirs(ruta, exist_ok=True) # Crea la carpeta si no existe

        fecha_hora: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Limpiar nombre_base_modelo por si viene de Enum.name
        nombre_limpio: str = nombre_base_modelo.replace("Modelo.", "")
        nombre_archivo: str = f"modelo_entrenado_{nombre_limpio}_{simbolo}_{fecha_hora}.joblib"
        ruta_final: str = os.path.join(ruta, nombre_archivo)

        joblib.dump(model, ruta_final)
        logging.info(f"✅ Modelo guardado en '{ruta_final}'")
    except Exception as e:
        logging.error(f"❌ Error al guardar el modelo en '{ruta}': {e}", exc_info=True)


# Funcion para cargar un modelo entrenado
def cargar_modelo(ruta: str) -> Any:
    """
    Carga un modelo desde un archivo .joblib.

    Args:
        ruta: La ruta completa al archivo .joblib del modelo.

    Returns:
        El objeto del modelo cargado. Se usa Any ya que no se sabe el tipo exacto guardado.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta especificada.
        Exception: Si ocurre otro error durante la carga.
    """
    if os.path.exists(ruta):
        try:
            logging.info(f"Cargando modelo desde '{ruta}'...")
            modelo_cargado: Any = joblib.load(ruta)
            logging.info("✅ Modelo cargado correctamente.")
            return modelo_cargado
        except Exception as e:
            logging.error(f"❌ Error al cargar el modelo desde '{ruta}': {e}", exc_info=True)
            # Relanzar la excepción o manejarla según sea necesario
            raise e # O retornar None, o lanzar una excepción personalizada
    else:
        logging.error(f"Modelo no encontrado en la ruta especificada: '{ruta}'")
        raise FileNotFoundError(f"No existe el modelo en '{ruta}'.")


# Funcion para crear o cargar, y luego entrenar un modelo
def crear_entrenar_modelo(
    ruta_modelo: Optional[str], # La ruta puede ser None si queremos crear uno nuevo
    simbolo: str,
    temporalidad: str,
    periodo: str,
    modelo_enum: Modelo = Modelo.RANDOM_FOREST, # Tipo de modelo a crear si no se carga
    target_method: TargetMethod = TargetMethod.ORIGINAL, # Pasar método de target a entrenar_modelo
    target_params: dict = {} # Pasar params de target a entrenar_modelo
) -> Optional[Any]: # Retorna el modelo entrenado o None si falla
    """
    Carga un modelo existente o crea uno nuevo, lo entrena con datos frescos,
    y lo guarda.

    Args:
        ruta_modelo: Ruta al archivo .joblib del modelo a cargar, o None para crear uno nuevo.
        simbolo: Símbolo para obtener datos y nombrar el modelo guardado.
        temporalidad: Intervalo de velas para obtener datos.
        periodo: Periodo histórico para obtener datos.
        modelo_enum: El tipo de modelo (del Enum Modelo) a crear si ruta_modelo es None.
        target_method: Método para definir el target ('original', 'atr', etc.)
        target_params: Parámetros para la función de definición de target.


    Returns:
        El modelo entrenado (tipo Any), o None si ocurre un error crítico.
    """
    model: Any # Usar Any ya que puede venir de cargar_modelo o generar_modelo_nuevo
    try:
        if ruta_modelo is None:
            logging.info("No se proporcionó ruta, generando nuevo modelo...")
            model = generar_modelo_nuevo(modelo_enum=modelo_enum)
        else:
            try:
                logging.info(f"Intentando cargar modelo desde: {ruta_modelo}")
                model = cargar_modelo(ruta_modelo)
            except FileNotFoundError:
                logging.warning("Modelo no encontrado en la ruta, generando nuevo modelo...")
                model = generar_modelo_nuevo(modelo_enum=modelo_enum)
            except Exception as e_load: # Capturar otros errores de carga
                 logging.error(f"Error inesperado al cargar modelo: {e_load}. Abortando.")
                 return None # No continuar si la carga falla inesperadamente

        # Obtener datos (asumiendo que obtener_datos_binance retorna pd.DataFrame)
        logging.info(f"Obteniendo datos para {simbolo} ({temporalidad}, {periodo})...")
        df: pd.DataFrame = obtener_datos_binance(simbolo=simbolo, intervalo=temporalidad, periodo=periodo)

        if df.empty:
             logging.error("No se pudieron obtener datos para el entrenamiento. Abortando.")
             return None # No continuar sin datos

        # Entrenar modelo (asumiendo que entrenar_modelo retorna el modelo entrenado o None si falla)
        logging.info("Entrenando modelo...")
        # Pasar método y params de target
        model_entrenado: Optional[Any] = entrenar_modelo(
             model,
             df,
             target_method=target_method,
             target_params=target_params
        )

        if model_entrenado is None:
            logging.error("El entrenamiento del modelo falló. Abortando.")
            return None

        # Guardar modelo entrenado
        logging.info("Guardando modelo entrenado...")
        # Usar el nombre del Enum para guardar si es posible
        nombre_base: str = getattr(modelo_enum, 'name', 'desconocido').replace("Modelo.", "")
        guardar_modelo(model_entrenado, nombre_base, simbolo)
        logging.info("✅ Modelo entrenado y guardado correctamente.")

        return model_entrenado

    except Exception as e:
        logging.error(f"❌ Ocurrió un error general en crear_entrenar_modelo: {e}", exc_info=True)
        return None # Retornar None en caso de cualquier error no capturado antes