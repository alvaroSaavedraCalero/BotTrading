import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

from config.config import TARGET_THRESHOLD, RANDOM_STATE


# Funcion para entrenar un modelo con los datos, adaptada para series temporales
def entrenar_modelo(model, df):
    """
    Entrena un modelo de clasificación usando datos históricos, adaptado para series temporales.

    Args:
        model: Instancia del modelo de scikit-learn (o compatible) a entrenar.
        df (pd.DataFrame): DataFrame con los datos históricos (OHLCV, etc.).

    Returns:
        Modelo entrenado (best_estimator_ de GridSearchCV).
    """
    logging.info('Iniciando entrenamiento de modelo (versión adaptada para series temporales)')

    # --- 1. Preparación de Datos y Características ---
    df = definir_target(df) # Asegúrate que esta función exista y defina 'target'

    df['returns'] = df['close'].pct_change()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = MACD(df['close']).macd_diff()
    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    # Eliminar filas con NaNs introducidos por indicadores/returns
    df = df.dropna()

    # Opcional: Filtrar clases si es necesario (evalúa si mejora tu caso)
    # df = df[df['target'].isin([0, 2])]
    # if df.empty:
    #     logging.error("No quedan datos después de filtrar NaNs y/o clases.")
    #     return None

    # Definir características y objetivo
    # Nota: Considera si 'temporalidad' debe ser tratada de otra forma (ej. one-hot encoding) si usas múltiples intervalos.
    # Si siempre usas el mismo intervalo, esta columna será constante y no aportará información al modelo.
    features_cols = ['open', 'high', 'low', 'close', 'temporalidad', 'returns', 'rsi', 'macd', 'ema9', 'ema21', 'atr']
    features = df[features_cols].astype(float)
    target = df['target']

    if features.empty or target.empty:
         logging.error("Features o Target están vacíos después del preprocesamiento.")
         return None

    # --- 2. División de Datos (Respetando Series Temporales) ---
    n_splits_cv = 5 # Número de divisiones para TimeSeriesSplit y GridSearchCV/cross_val_score
    tscv = TimeSeriesSplit(n_splits=n_splits_cv)

    # Dividir en entrenamiento y prueba final (hold-out set) cronológicamente
    # Usaremos todas las divisiones de tscv para el entrenamiento/validación cruzada,
    # y reservaremos la última parte como conjunto de prueba final si es necesario,
    # aunque GridSearchCV ya hace validación interna.
    # Una forma común es entrenar/validar con tscv y luego evaluar en datos futuros no vistos.
    # Para simplificar aquí, usaremos tscv dentro de GridSearchCV y cross_val_score directamente sobre 'features' y 'target'.
    # Otra opción es separar manualmente:
    test_size_ratio = 0.2 # Porcentaje para el conjunto de prueba final
    split_index = int(len(features) * (1 - test_size_ratio))
    X_train, X_test = features[:split_index], features[split_index:]
    y_train, y_test = target[:split_index], target[split_index:]

    if X_train.empty or X_test.empty:
        logging.error("Conjuntos de entrenamiento o prueba vacíos después de la división temporal.")
        return None

    logging.info(f"División temporal: Entrenamiento={len(X_train)} muestras, Prueba={len(X_test)} muestras.")


    # --- 3. Manejo de Desbalance de Clases (Calculado sobre el conjunto de entrenamiento) ---
    clases = np.unique(y_train)
    if len(clases) < 2:
        logging.warning(f"Solo se encontró {len(clases)} clase(s) en el conjunto de entrenamiento. El entrenamiento podría no ser efectivo.")
        # Podrías decidir detenerte o continuar con precaución
        # return None # Opcional: detener si no hay suficientes clases

    pesos = compute_class_weight(class_weight='balanced', classes=clases, y=y_train)
    class_weights_dict = {clase: peso for clase, peso in zip(clases, pesos)}
    logging.info(f"Pesos de clase calculados (entrenamiento): {class_weights_dict}")


    # --- 4. Configuración y Búsqueda de Hiperparámetros con GridSearchCV y TimeSeriesSplit ---
    param_grid = {} # Inicializar param_grid
    model_name = model.__class__.__name__
    fit_params = {} # Parámetros adicionales para el método fit (como sample_weight)

    # Ajustar el grid y parámetros específicos del modelo
    if model_name == 'RandomForestClassifier':
        model.set_params(class_weight=class_weights_dict, random_state=RANDOM_STATE) # Añadir random_state si es aplicable
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7, None], # Añadido None
            'max_features': ['sqrt', 'log2'], # Cambiado 'auto' a opciones explícitas
            'min_samples_split': [2, 5], # Añadido
            'min_samples_leaf': [1, 3]   # Añadido
        }
    elif model_name == 'GradientBoostingClassifier':
        model.set_params(random_state=RANDOM_STATE) # Añadir random_state si es aplicable
        # GradientBoosting no usa class_weight directamente, usamos sample_weight en fit
        fit_params['sample_weight'] = y_train.map(class_weights_dict).fillna(1.0).values # Asegurar que todos los y_train tengan peso
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0] # Añadido
        }
    elif model_name == 'XGBClassifier':
        # XGBoost puede usar scale_pos_weight para binario o sample_weight para multiclase
        # Calcularemos sample_weight para el caso general multiclase
        model.set_params(random_state=RANDOM_STATE) # Añadir random_state si es aplicable
        fit_params['sample_weight'] = y_train.map(class_weights_dict).fillna(1.0).values
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0], # Añadido
            'colsample_bytree': [0.8, 1.0] # Añadido
        }
    else:
        logging.error(f"Modelo '{model_name}' no soportado en tuning automático configurado.")
        raise ValueError(f"Modelo '{model_name}' no soportado en tuning automático")

    # Ejecutar GridSearchCV usando TimeSeriesSplit y métrica F1 ponderada
    # Usamos el conjunto de entrenamiento (X_train, y_train) para la búsqueda
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               scoring='f1_weighted', # *** Métrica cambiada a f1_weighted ***
                               cv=tscv, # *** Usando TimeSeriesSplit para validación cruzada ***
                               n_jobs=-1, # Usar todos los cores disponibles
                               verbose=1) # Mostrar progreso

    try:
        grid_search.fit(X_train, y_train, **fit_params) # Pasar sample_weight si es necesario
        logging.info(f"Mejores parámetros encontrados: {grid_search.best_params_}")
        logging.info(f"Mejor puntuación F1 ponderada (CV en entreno): {grid_search.best_score_:.4f}")
        best_model = grid_search.best_estimator_

    except Exception as e:
        logging.error(f"Error durante GridSearchCV: {e}")
        return None


    # --- 5. Evaluación Final (sobre el conjunto de prueba Hold-Out) ---
    # Re-entrenar el mejor modelo con TODO el conjunto de entrenamiento (GridSearchCV puede hacerlo con refit=True por defecto)
    # best_model = grid_search.best_estimator_ # Ya asignado arriba

    # Evaluar en el conjunto de prueba que se mantuvo separado cronológicamente
    y_pred = best_model.predict(X_test)
    logging.info("--- Evaluación en Conjunto de Prueba (Hold-Out) ---")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")


    # --- Opcional: Cross-Validation Score sobre el conjunto de entrenamiento ---
    # Para tener una idea de la estabilidad del modelo en el tiempo ANTES del test set
    try:
        # Usamos X_train, y_train para esta evaluación, con la misma división temporal
        scores = cross_val_score(best_model, X_train, y_train, cv=tscv, scoring='f1_weighted', fit_params=fit_params)
        logging.info(f"Cross-Validation F1-Weighted Scores (en entreno): {scores}")
        logging.info(f"✅ Modelo entrenado. Media CV F1 (entreno): {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    except Exception as e:
        logging.warning(f"No se pudo calcular cross_val_score: {e}")


    logging.info('✅ Entrenamiento de modelo finalizado.')
    return best_model 


# Funcion para obtener el target del entrenamiento del modelo
def definir_target(df):
    umbral = TARGET_THRESHOLD  
    df['returns'] = df['close'].pct_change()
    df['target'] = df['returns'].shift(-1).apply(
        lambda x: 2 if x > umbral else (0 if x < -umbral else 1)
    )
    df.dropna(inplace=True)
    return df




