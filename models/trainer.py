# models/trainer.py

import logging
import numpy as np
import pandas as pd # Importar pandas
from typing import List, Dict, Any, Optional, Tuple # Importar tipos

# Importar clases y funciones específicas de sklearn
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator # Para tipar el modelo genérico
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier # Para tipar en GridSearch (opcional)
from xgboost import XGBClassifier # Para tipar en GridSearch (opcional)

# Importar librerías TA
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

# Importar Enum y config
from utils.enumerados import TargetMethod # Asumiendo que TargetMethod es un Enum
from config.config import TARGET_THRESHOLD, RANDOM_STATE

# --- Definiciones de Funciones para el Target ---

def definir_target(df: pd.DataFrame, umbral: float = TARGET_THRESHOLD) -> pd.DataFrame:
    """
    Define el target original basado en el retorno de la siguiente vela y un umbral fijo.
    Target: 2 (sube), 0 (baja), 1 (neutral).

    Args:
        df: DataFrame de entrada.
        umbral: Umbral para definir subida/bajada.

    Returns:
        DataFrame con la columna 'target' añadida/modificada.
    """
    logging.debug(f"Definiendo target original con umbral fijo: {umbral}")
    df_copy = df.copy() # Trabajar con copia
    if 'returns' not in df_copy.columns:
        df_copy['returns'] = df_copy['close'].pct_change()

    # Usar apply en la Serie directamente
    next_returns: pd.Series = df_copy['returns'].shift(-1)
    df_copy['target'] = next_returns.apply(
        lambda x: 2 if x > umbral else (0 if x < -umbral else 1)
    )
    # Marcar NaNs introducidos por shift para eliminarlos después
    df_copy['target'] = df_copy['target'].where(next_returns.notna())

    return df_copy

def definir_target_dinamico_atr(df: pd.DataFrame, atr_period: int = 14, atr_multiplier: float = 0.5) -> pd.DataFrame:
    """
    Define el target usando un umbral dinámico basado en ATR.
    Target: 2 (sube), 0 (baja), 1 (neutral).

    Args:
        df: DataFrame de entrada.
        atr_period: Periodo para calcular el ATR.
        atr_multiplier: Multiplicador del ATR para definir el umbral.

    Returns:
        DataFrame con la columna 'target' añadida/modificada.
    """
    logging.debug(f"Definiendo target dinámico ATR con periodo={atr_period}, multiplicador={atr_multiplier}")
    df_copy = df.copy()
    # Calcular ATR
    df_copy['atr'] = AverageTrueRange(df_copy['high'], df_copy['low'], df_copy['close'], window=atr_period).average_true_range()
    # Calcular el retorno del siguiente periodo
    df_copy['next_return'] = df_copy['close'].pct_change().shift(-1)
    # Definir umbral dinámico
    df_copy['dynamic_threshold'] = df_copy['atr'] * atr_multiplier

    # Definir target usando numpy.select
    # Asegurarse que las columnas existen y no tienen NaNs inesperados antes de comparar
    mask_valid: pd.Series = df_copy['next_return'].notna() & df_copy['dynamic_threshold'].notna()
    conditions: List[pd.Series] = [
        df_copy['next_return'] > df_copy['dynamic_threshold'],
        df_copy['next_return'] < -df_copy['dynamic_threshold']
    ]
    choices: List[int] = [2, 0]
    # Aplicar np.select solo donde los datos son válidos, marcar otros como NaN
    df_copy['target'] = np.where(mask_valid, np.select(conditions, choices, default=1), np.nan)

    return df_copy

def definir_target_horizonte_n(df: pd.DataFrame, n_periods: int = 5, umbral: float = TARGET_THRESHOLD) -> pd.DataFrame:
    """
    Define el target basado en el retorno acumulado sobre los próximos n_periods.
    Target: 2 (sube), 0 (baja), 1 (neutral).

    Args:
        df: DataFrame de entrada.
        n_periods: Número de periodos hacia adelante.
        umbral: Umbral de retorno.

    Returns:
        DataFrame con la columna 'target' añadida/modificada.
    """
    logging.debug(f"Definiendo target horizonte N={n_periods} con umbral={umbral}")
    df_copy = df.copy()
    # Calcular retorno futuro
    df_copy['future_return_n'] = df_copy['close'].pct_change(periods=n_periods).shift(-n_periods)
    # Definir target
    future_returns_series: pd.Series = df_copy['future_return_n']
    df_copy['target'] = future_returns_series.apply(
        lambda x: 2 if x > umbral else (0 if x < -umbral else 1)
    )
    # Marcar NaNs
    df_copy['target'] = df_copy['target'].where(future_returns_series.notna())
    return df_copy

def definir_target_nivel_alcanzado(df: pd.DataFrame, n_periods: int = 5, umbral_ratio: float = TARGET_THRESHOLD * 1.5) -> pd.DataFrame:
    """
    Define el target basado en si el high/low futuro cruza un umbral relativo al cierre actual.
    Target: 2 (nivel up), 0 (nivel down), 1 (ninguno).

    Args:
        df: DataFrame de entrada.
        n_periods: Ventana futura a considerar.
        umbral_ratio: Ratio sobre cierre actual para niveles.

    Returns:
        DataFrame con la columna 'target' añadida/modificada.
    """
    logging.debug(f"Definiendo target nivel alcanzado N={n_periods} con ratio={umbral_ratio}")
    df_copy = df.copy()
    # Calcular niveles objetivo
    df_copy['target_up_level'] = df_copy['close'] * (1 + umbral_ratio)
    df_copy['target_down_level'] = df_copy['close'] * (1 - umbral_ratio)
    # Encontrar max/min futuro
    df_copy['future_max_high'] = df_copy['high'].shift(-n_periods).rolling(window=n_periods, min_periods=1).max()
    df_copy['future_min_low'] = df_copy['low'].shift(-n_periods).rolling(window=n_periods, min_periods=1).min()
    # Determinar si los niveles fueron alcanzados
    df_copy['hit_up'] = df_copy['future_max_high'] >= df_copy['target_up_level']
    df_copy['hit_down'] = df_copy['future_min_low'] <= df_copy['target_down_level']
    # Asignar target
    mask_valid: pd.Series = df_copy['future_max_high'].notna() & df_copy['future_min_low'].notna()
    conditions: List[pd.Series] = [df_copy['hit_up'], df_copy['hit_down']]
    choices: List[int] = [2, 0]
    df_copy['target'] = np.where(mask_valid, np.select(conditions, choices, default=1), np.nan)
    return df_copy


# --- Función Principal de Entrenamiento ---

# Usar un tipo genérico para el modelo o Union si se prefiere
ModelInputType = Any # O Union[GradientBoostingClassifier, RandomForestClassifier, XGBClassifier, BaseEstimator]

def entrenar_modelo(
    model: ModelInputType,
    df_input: pd.DataFrame,
    target_method: TargetMethod = TargetMethod.ORIGINAL, # Usar el Enum directamente
    target_params: Dict[str, Any] = {}
) -> Optional[Any]: # El modelo entrenado (best_estimator_) o None
    """
    Entrena un modelo de clasificación usando datos históricos, adaptado para series temporales,
    y permitiendo diferentes métodos para definir el target. Los mismos ``fit_params`` empleados
    en ``GridSearchCV`` se pasan opcionalmente a ``cross_val_score`` para evaluar con
    ponderaciones (por ejemplo, ``sample_weight``) coherentes en cada fold.

    Args:
        model: Instancia del modelo (scikit-learn, xgboost, etc.) a entrenar.
        df_input: DataFrame con los datos históricos (OHLCV, 'temporalidad', etc.).
        target_method: Método (del Enum TargetMethod) a usar para definir el target.
        target_params: Diccionario con parámetros para la función de definición de target.

    Returns:
        Modelo entrenado (best_estimator_ de GridSearchCV) o None si falla.
    """
    logging.info(f"Iniciando entrenamiento de modelo. Usando método target: '{target_method.name}'") # Usar .name para el log
    df: pd.DataFrame = df_input.copy() # Trabajar con una copia

    # --- 1. Definición del Target ---
    logging.info(f"Aplicando método target '{target_method.name}' con parámetros: {target_params}")
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
    logging.info("Calculando características adicionales (RSI, MACD, EMAs, Returns)...")
    if 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()
    if 'atr' not in df.columns:
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    # No añadir tipos :pd.Series aquí
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = MACD(df['close']).macd_diff()
    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()

    # --- 3. Limpieza Final de Datos ---
    initial_rows: int = len(df)
    df = df.dropna(subset=['target']) # Asegura que el target calculado sea válido
    features_cols: List[str] = ['open', 'high', 'low', 'close', 'temporalidad', 'returns', 'rsi', 'macd', 'ema9', 'ema21', 'atr']
    # Asegurarse que todas las features existan antes de dropna
    missing_cols = [col for col in features_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Faltan columnas de features requeridas: {missing_cols}. Abortando.")
        return None
    df = df.dropna(subset=features_cols) # Eliminar NaNs de las features
    final_rows: int = len(df)
    logging.info(f"Filas eliminadas por NaNs (target o features): {initial_rows - final_rows}. Filas restantes: {final_rows}")

    if df.empty:
        logging.error("El DataFrame está vacío después de calcular features y eliminar NaNs.")
        return None

    # Definir características y objetivo finales
    features: pd.DataFrame = df[features_cols].astype(float)
    target: pd.Series = df['target'].astype(int) # Target es una Serie de pandas

    if features.empty or target.empty:
         logging.error("Features o Target están vacíos después del preprocesamiento final.")
         return None
    if len(target.unique()) < 2:
         logging.warning(f"Solo hay {len(target.unique())} clase(s) en los datos finales. El modelo podría no aprender correctamente.")

    # --- 4. División de Datos ---
    n_splits_cv: int = 5
    tscv: TimeSeriesSplit = TimeSeriesSplit(n_splits=n_splits_cv)
    test_size_ratio: float = 0.2
    split_index: int = int(len(features) * (1 - test_size_ratio))
    X_train: pd.DataFrame = features[:split_index]
    X_test: pd.DataFrame = features[split_index:]
    y_train: pd.Series = target[:split_index] # y_train/y_test son Series
    y_test: pd.Series = target[split_index:]

    if X_train.empty or X_test.empty:
        logging.error("Conjuntos de entrenamiento o prueba vacíos después de la división temporal.")
        return None
    if len(y_train.unique()) < 2:
        logging.warning(f"El conjunto de entrenamiento solo tiene {len(y_train.unique())} clase(s).")

    logging.info(f"División temporal: Entrenamiento={len(X_train)} muestras, Prueba={len(X_test)} muestras.")

    # --- 5. Manejo de Desbalance de Clases ---
    clases: np.ndarray = np.unique(y_train) # unique devuelve ndarray
    pesos: np.ndarray = compute_class_weight(class_weight='balanced', classes=clases, y=y_train) # pesos es ndarray
    # El tipo de las claves depende de 'clases', podría ser np.int64. Usar Any o tipo específico.
    class_weights_dict: Dict[Any, float] = {clase: peso for clase, peso in zip(clases, pesos)}
    logging.info(f"Pesos de clase calculados (entrenamiento): {class_weights_dict}")

    # --- 6. Configuración y Búsqueda de Hiperparámetros ---
    param_grid: Dict[str, List[Any]] = {} # Grid de parámetros
    model_name: str = model.__class__.__name__
    fit_params: Dict[str, Any] = {} # Params para .fit (ej. sample_weight)

    logging.info(f"Configurando GridSearchCV para el modelo: {model_name}")
    # Ajustar grid y parámetros específicos
    # (El código para configurar param_grid y fit_params según model_name se omite por brevedad,
    # pero debería tener tipos internos si se desea, ej. List[int], List[float], etc.)
    # ... (código de configuración de param_grid y fit_params idéntico al anterior) ...
    if model_name == 'RandomForestClassifier':
        model.set_params(class_weight=class_weights_dict, random_state=RANDOM_STATE, n_jobs=-1)
        param_grid = {
             'n_estimators': [100, 300], 'max_depth': [5, 7, None], 'max_features': ['sqrt', 'log2'],
             'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 3, 5]
        }
    elif model_name == 'GradientBoostingClassifier':
        model.set_params(random_state=RANDOM_STATE)
        # sample_weight debe ser un ndarray
        fit_params['sample_weight'] = y_train.map(class_weights_dict).fillna(1.0).values
        param_grid = {
             'n_estimators': [100, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1],
             'subsample': [0.8, 1.0]
        }
    elif model_name == 'XGBClassifier':
         num_class: int = len(clases)
         objective: str = 'multi:softmax' if num_class > 2 else 'binary:logistic'
         model.set_params(objective=objective, num_class=num_class if num_class > 2 else None, # num_class solo para multi
                          random_state=RANDOM_STATE, n_jobs=-1)
         fit_params['sample_weight'] = y_train.map(class_weights_dict).fillna(1.0).values
         param_grid = {
             'n_estimators': [100, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1, 0.2],
             'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]
         }
    else:
        logging.error(f"Modelo '{model_name}' no soportado en tuning automático configurado.")
        raise ValueError(f"Modelo '{model_name}' no soportado en tuning automático")


    logging.info(f"Iniciando GridSearchCV con {n_splits_cv}-fold TimeSeriesSplit y scoring 'f1_weighted'.")
    # Tipar grid_search como GridSearchCV
    grid_search: GridSearchCV = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_weighted',
                                            cv=tscv, n_jobs=-1, verbose=1)

    try:
        grid_search.fit(X_train, y_train, **fit_params)
        logging.info(f"Mejores parámetros encontrados por GridSearchCV: {grid_search.best_params_}")
        logging.info(f"Mejor puntuación F1 ponderada (CV en entreno): {grid_search.best_score_:.4f}")
        # best_estimator_ devuelve el tipo base del estimador, usar Any o ModelInputType
        best_model: Any = grid_search.best_estimator_

    except Exception as e:
        logging.exception(f"Error durante GridSearchCV: {e}")
        return None

    # --- 7. Evaluación Final ---
    logging.info("--- Evaluación Final en Conjunto de Prueba (Hold-Out) ---")
    try:
        # predict devuelve ndarray
        y_pred: np.ndarray = best_model.predict(X_test)
        # unique devuelve ndarray
        labels_test: np.ndarray = np.unique(y_test)
        labels_pred: np.ndarray = np.unique(y_pred)
        # combine unique labels, sort them, convert to list of int/float/str as needed
        combined_labels: List[Any] = sorted(list(set(labels_test) | set(labels_pred)))

        logging.info(f"Classification Report (Test Set):\n"
                     f"{classification_report(y_test, y_pred, labels=combined_labels, zero_division=0)}")
        logging.info(f"Confusion Matrix (Test Set):\n"
                     f"{confusion_matrix(y_test, y_pred, labels=combined_labels)}")
    except Exception as e:
        logging.exception(f"Error durante la evaluación final en el conjunto de prueba: {e}")


    # --- 8. Opcional: Cross-Validation Score ---
    logging.info("--- Cross-Validation Opcional sobre Conjunto de Entrenamiento ---")
    # cross_val_score acepta fit_params; se reutilizan los mismos parámetros
    # (por ejemplo, sample_weight) empleados en GridSearchCV
    if fit_params:
         cv_params = fit_params
    else:
         cv_params = None  # Pasar None si está vacío

    try:
        # scores es ndarray
        scores: np.ndarray = cross_val_score(
            best_model,
            X_train,
            y_train,
            cv=tscv,
            scoring='f1_weighted',
            fit_params=cv_params,
            n_jobs=-1,
        )
        logging.info(
            f"Cross-Validation F1-Weighted Scores (en entreno): {scores}"
        )
        logging.info(
            f"Media CV F1 (entreno): {scores.mean():.4f} (+/- {scores.std() * 2:.4f})"
        )
    except Exception as e:
        logging.warning(f"No se pudo calcular cross_val_score opcional: {e}", exc_info=True)


    logging.info('✅ Entrenamiento de modelo finalizado.')
    return best_model