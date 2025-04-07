# models/trainer.py

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

from utils.enumerados import TargetMethod

from config.config import TARGET_THRESHOLD, RANDOM_STATE

# --- Definiciones de Funciones para el Target --- 

def definir_target(df, umbral=TARGET_THRESHOLD):
    """
    Define el target original basado en el retorno de la siguiente vela y un umbral fijo.
    Target: 2 (sube), 0 (baja), 1 (neutral).
    """
    logging.debug(f"Definiendo target original con umbral fijo: {umbral}")
    # Calcular returns si no existe (aunque se recalculará en entrenar_modelo)
    if 'returns' not in df.columns:
         df['returns'] = df['close'].pct_change()

    df['target'] = df['returns'].shift(-1).apply(
        lambda x: 2 if x > umbral else (0 if x < -umbral else 1)
    )
    # No eliminar NaNs aquí, se hará centralizadamente después en entrenar_modelo
    # df.dropna(inplace=True)
    return df

def definir_target_dinamico_atr(df, atr_period=14, atr_multiplier=0.5):
    """
    Define el target usando un umbral dinámico basado en ATR.
    Target: 2 (sube), 0 (baja), 1 (neutral).
    """
    logging.debug(f"Definiendo target dinámico ATR con periodo={atr_period}, multiplicador={atr_multiplier}")
    # 1. Calcular ATR
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=atr_period).average_true_range()

    # 2. Calcular el retorno del siguiente periodo
    df['next_return'] = df['close'].pct_change().shift(-1)

    # 3. Definir umbral dinámico para cada fila (se usará después de dropna)
    df['dynamic_threshold'] = df['atr'] * atr_multiplier

    # 4. Definir target usando numpy.select para claridad
    # Nota: Las condiciones se evaluarán después de eliminar NaNs de las columnas necesarias
    conditions = [
        df['next_return'] > df['dynamic_threshold'],  # Condición subida
        df['next_return'] < -df['dynamic_threshold'] # Condición bajada
    ]
    choices = [2, 0]
    df['target'] = np.select(conditions, choices, default=1) # Default: neutral

    # Marcar filas con NaN en columnas críticas para eliminar después
    df['target'] = np.where(df['atr'].isna() | df['next_return'].isna(), np.nan, df['target'])

    # 5. Opcional: Eliminar columnas auxiliares si no se necesitan después (mejor no hacerlo aquí)
    # df = df.drop(columns=['next_return', 'dynamic_threshold'])

    return df

def definir_target_horizonte_n(df, n_periods=5, umbral=TARGET_THRESHOLD):
    """
    Define el target basado en el retorno acumulado sobre los próximos n_periods.
    Target: 2 (sube), 0 (baja), 1 (neutral).
    """
    logging.debug(f"Definiendo target horizonte N={n_periods} con umbral={umbral}")
    # 1. Calcular retorno sobre los próximos N periodos
    df['future_return_n'] = df['close'].pct_change(periods=n_periods).shift(-n_periods)

    # 2. Definir target basado en el retorno futuro
    df['target'] = df['future_return_n'].apply(
        lambda x: 2 if x > umbral else (0 if x < -umbral else 1)
    )
    # Marcar filas con NaN para eliminar después
    df['target'] = np.where(df['future_return_n'].isna(), np.nan, df['target'])

    # 3. Opcional: Eliminar columna auxiliar
    # df = df.drop(columns=['future_return_n'])

    return df

def definir_target_nivel_alcanzado(df, n_periods=5, umbral_ratio=TARGET_THRESHOLD * 1.5):
    """
    Define el target basado en si el high/low futuro cruza un umbral relativo al cierre actual.
    Target: 2 (nivel up alcanzado), 0 (nivel down alcanzado), 1 (ninguno).
    """
    logging.debug(f"Definiendo target nivel alcanzado N={n_periods} con ratio={umbral_ratio}")
    # Calcular niveles objetivo basados en el cierre actual
    df['target_up_level'] = df['close'] * (1 + umbral_ratio)
    df['target_down_level'] = df['close'] * (1 - umbral_ratio)

    # Encontrar el máximo high y mínimo low en las próximas N velas
    df['future_max_high'] = df['high'].shift(-n_periods).rolling(window=n_periods, min_periods=1).max()
    df['future_min_low'] = df['low'].shift(-n_periods).rolling(window=n_periods, min_periods=1).min()

    # Determinar si los niveles fueron alcanzados
    df['hit_up'] = df['future_max_high'] >= df['target_up_level']
    df['hit_down'] = df['future_min_low'] <= df['target_down_level']

    # Asignar target
    conditions = [
        df['hit_up'],   # Si se alcanza el nivel superior -> Target 2
        df['hit_down']  # Si se alcanza el nivel inferior -> Target 0
    ]
    choices = [2, 0]
    df['target'] = np.select(conditions, choices, default=1) # Si no se alcanza ninguno -> Target 1

    # Marcar filas con NaN para eliminar después
    df['target'] = np.where(df['future_max_high'].isna() | df['future_min_low'].isna(), np.nan, df['target'])

    # Opcional: Eliminar columnas auxiliares
    # df = df.drop(columns=['target_up_level', 'target_down_level', 'future_max_high', 'future_min_low', 'hit_up', 'hit_down'])

    return df



# --- Función Principal de Entrenamiento ---

def entrenar_modelo(model, df_input, target_method=TargetMethod.ORIGINAL, target_params={}):
    """
    Entrena un modelo de clasificación usando datos históricos, adaptado para series temporales,
    y permitiendo diferentes métodos para definir el target.

    Args:
        model: Instancia del modelo de scikit-learn (o compatible) a entrenar.
        df_input (pd.DataFrame): DataFrame con los datos históricos (OHLCV, 'temporalidad', etc.).
        target_method (str): Método a usar para definir el target ('original', 'atr', 'horizonte_n', 'nivel_alcanzado').
        target_params (dict): Diccionario con parámetros para la función de definición de target.

    Returns:
        Modelo entrenado (best_estimator_ de GridSearchCV) o None si falla.
    """
    logging.info(f"Iniciando entrenamiento de modelo. Usando método target: '{target_method}'")
    df = df_input.copy() # Trabajar con una copia para no modificar el original

    # --- 1. Definición del Target ---
    logging.info(f"Aplicando método target '{target_method}' con parámetros: {target_params}")
    if target_method == TargetMethod.ORIGINAL:
        df = definir_target(df, **target_params)
    elif target_method == TargetMethod.ATR:
        df = definir_target_dinamico_atr(df, **target_params)
    elif target_method == TargetMethod.HORIZONTE_N:
        df = definir_target_horizonte_n(df, **target_params)
    elif target_method == TargetMethod.NIVEL_ALCANZADO:
        df = definir_target_nivel_alcanzado(df, **target_params)
    else:
        logging.error(f"Método de target '{target_method}' no reconocido.")
        raise ValueError(f"Método de target '{target_method}' no reconocido.")

    # --- 2. Ingeniería de Características Adicionales ---
    # Calcular features básicos necesarios para el modelo que no hayan sido calculados por la func target
    logging.info("Calculando características adicionales (RSI, MACD, EMAs, Returns)...")
    if 'returns' not in df.columns: # Calcular si no existe (target original lo hace, otros no necesariamente)
         df['returns'] = df['close'].pct_change()
    if 'atr' not in df.columns: # Calcular ATR si no se usó el método dinámico
         df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range() # Usar un ATR_PERIOD de config si se prefiere

    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = MACD(df['close']).macd_diff()
    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()

    # --- 3. Limpieza Final de Datos ---
    # Eliminar filas con NaN introducidos por indicadores Y por la función target
    # Es crucial eliminar filas donde el target sea NaN (porque no se pudo calcular)
    initial_rows = len(df)
    df = df.dropna(subset=['target']) # Asegura que el target es válido
    # Eliminar NaNs de las features que usará el modelo
    features_cols = ['open', 'high', 'low', 'close', 'temporalidad', 'returns', 'rsi', 'macd', 'ema9', 'ema21', 'atr']
    df = df.dropna(subset=features_cols)
    final_rows = len(df)
    logging.info(f"Filas eliminadas por NaNs (target o features): {initial_rows - final_rows}. Filas restantes: {final_rows}")

    if df.empty:
        logging.error("El DataFrame está vacío después de calcular features y eliminar NaNs.")
        return None

    # Opcional: Filtrar clases si es necesario (evalúa si mejora tu caso)
    # df = df[df['target'].isin([0, 2])]
    # if df.empty:
    #     logging.error("No quedan datos después de filtrar NaNs y/o clases.")
    #     return None

    # Definir características y objetivo finales
    features = df[features_cols].astype(float)
    target = df['target'].astype(int) # Asegurar que target es entero para clasificación

    if features.empty or target.empty:
         logging.error("Features o Target están vacíos después del preprocesamiento final.")
         return None
    if len(target.unique()) < 2:
         logging.warning(f"Solo hay {len(target.unique())} clase(s) en los datos finales. El modelo podría no aprender correctamente.")
         # Podrías retornar None aquí si lo consideras un error crítico

    # --- 4. División de Datos (Respetando Series Temporales) ---
    n_splits_cv = 5 # Número de divisiones para TimeSeriesSplit y GridSearchCV/cross_val_score
    tscv = TimeSeriesSplit(n_splits=n_splits_cv)

    test_size_ratio = 0.2 # Porcentaje para el conjunto de prueba final (hold-out)
    split_index = int(len(features) * (1 - test_size_ratio))
    X_train, X_test = features[:split_index], features[split_index:]
    y_train, y_test = target[:split_index], target[split_index:]

    if X_train.empty or X_test.empty:
        logging.error("Conjuntos de entrenamiento o prueba vacíos después de la división temporal.")
        return None
    if len(y_train.unique()) < 2:
        logging.warning(f"El conjunto de entrenamiento solo tiene {len(y_train.unique())} clase(s).")
        # Considera retornar None si no hay suficientes clases para entrenar

    logging.info(f"División temporal: Entrenamiento={len(X_train)} muestras, Prueba={len(X_test)} muestras.")


    # --- 5. Manejo de Desbalance de Clases (Calculado sobre el conjunto de entrenamiento) ---
    clases = np.unique(y_train)
    pesos = compute_class_weight(class_weight='balanced', classes=clases, y=y_train)
    class_weights_dict = {clase: peso for clase, peso in zip(clases, pesos)}
    logging.info(f"Pesos de clase calculados (entrenamiento): {class_weights_dict}")


    # --- 6. Configuración y Búsqueda de Hiperparámetros con GridSearchCV y TimeSeriesSplit ---
    param_grid = {} # Inicializar param_grid
    model_name = model.__class__.__name__
    fit_params = {} # Parámetros adicionales para el método fit (como sample_weight)

    logging.info(f"Configurando GridSearchCV para el modelo: {model_name}")
    # Ajustar el grid y parámetros específicos del modelo
    if model_name == 'RandomForestClassifier':
        model.set_params(class_weight=class_weights_dict, random_state=RANDOM_STATE, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 300], #[100, 300, 500],
            'max_depth': [5, 7, None], #[3, 5, 7, None],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10], #[2, 5],
            'min_samples_leaf': [1, 3, 5] #[1, 3]
        }
    elif model_name == 'GradientBoostingClassifier':
        model.set_params(random_state=RANDOM_STATE)
        fit_params['sample_weight'] = y_train.map(class_weights_dict).fillna(1.0).values
        param_grid = {
            'n_estimators': [100, 300], #[100, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1], #[0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        }
    elif model_name == 'XGBClassifier':
         # Asegúrate de que XGBoost esté instalado.
         # XGBoost usa 'objective' y puede necesitar mapeo de clases si no son 0, 1, 2...
         num_class = len(clases)
         if num_class > 2:
             model.set_params(objective='multi:softmax', num_class=num_class, random_state=RANDOM_STATE, n_jobs=-1)
         else: # Asumir binario si solo quedan 2 clases (ej. si se filtra clase 1)
              model.set_params(objective='binary:logistic', random_state=RANDOM_STATE, n_jobs=-1)

         fit_params['sample_weight'] = y_train.map(class_weights_dict).fillna(1.0).values
         param_grid = {
             'n_estimators': [100, 300], #[100, 300, 500],
             'max_depth': [3, 5, 7],
             'learning_rate': [0.05, 0.1, 0.2], #[0.01, 0.05, 0.1],
             'subsample': [0.8, 1.0],
             'colsample_bytree': [0.8, 1.0]
         }
         # scale_pos_weight es solo para binario, sample_weight es más general
         # if num_class == 2 and 0 in class_weights_dict and 1 in class_weights_dict:
         #     model.set_params(scale_pos_weight = class_weights_dict[0] / class_weights_dict[1]) # O al revés dependiendo de cuál es la clase positiva

    else:
        logging.error(f"Modelo '{model_name}' no soportado en tuning automático configurado.")
        raise ValueError(f"Modelo '{model_name}' no soportado en tuning automático")

    # Ejecutar GridSearchCV
    logging.info(f"Iniciando GridSearchCV con {n_splits_cv}-fold TimeSeriesSplit y scoring 'f1_weighted'.")
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               scoring='f1_weighted', # Métrica robusta para desbalance
                               cv=tscv, # Validación cruzada temporal
                               n_jobs=-1, # Usar todos los cores
                               verbose=1)

    try:
        grid_search.fit(X_train, y_train, **fit_params) # Pasar sample_weight si es necesario
        logging.info(f"Mejores parámetros encontrados por GridSearchCV: {grid_search.best_params_}")
        logging.info(f"Mejor puntuación F1 ponderada (CV en entreno): {grid_search.best_score_:.4f}")
        best_model = grid_search.best_estimator_

    except Exception as e:
        logging.exception(f"Error durante GridSearchCV: {e}") # Usar logging.exception para traza completa
        return None

    # --- 7. Evaluación Final (sobre el conjunto de prueba Hold-Out) ---
    logging.info("--- Evaluación Final en Conjunto de Prueba (Hold-Out) ---")
    try:
        y_pred = best_model.predict(X_test)
        # Asegurarse que y_pred y y_test tengan las mismas clases posibles para el reporte
        labels = sorted(np.unique(np.concatenate((y_test, y_pred))))
        logging.info(f"Classification Report (Test Set):\n"
                     f"{classification_report(y_test, y_pred, labels=labels, zero_division=0)}")
        logging.info(f"Confusion Matrix (Test Set):\n"
                     f"{confusion_matrix(y_test, y_pred, labels=labels)}")
    except Exception as e:
        logging.exception(f"Error durante la evaluación final en el conjunto de prueba: {e}")


    # --- 8. Opcional: Cross-Validation Score sobre el conjunto de entrenamiento ---
    
    logging.info("--- Cross-Validation Opcional sobre Conjunto de Entrenamiento ---")
    try:
        # Usamos X_train, y_train para esta evaluación, con la misma división temporal
        # Puede ser útil si el best_score_ de GridSearch parece demasiado bueno o inestable
        scores = cross_val_score(best_model, X_train, y_train, cv=tscv, scoring='f1_weighted', params=fit_params, n_jobs=-1) # type: ignore
        logging.info(f"Cross-Validation F1-Weighted Scores (en entreno): {scores}")
        logging.info(f"Media CV F1 (entreno): {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    except Exception as e:
        logging.warning(f"No se pudo calcular cross_val_score opcional: {e}", exc_info=True)


    logging.info('✅ Entrenamiento de modelo finalizado.')
    return best_model
