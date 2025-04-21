# models/trainer.py

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
import os

# Importar clases y funciones específicas de sklearn
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# Importar nuestros propios módulos
from utils.indicators import prepare_features  # Nueva función que unifica la ingeniería de características
# Importar Enum y config
from utils.enumerados import TargetMethod
from config.config import TARGET_THRESHOLD, RANDOM_STATE, MODEL_SAVE_DIR
# --- Definiciones de Funciones para el Target ---

def definir_target(df: pd.DataFrame, method: TargetMethod, **params) -> pd.DataFrame:
    """
    Define target variable based on the specified method.
    
    Args:
        df: DataFrame with price data
        method: Method to use for target definition
        params: Additional parameters for the target method
    
    Returns:
        DataFrame with target column added
    """
    df_copy = df.copy()
    
    if method == TargetMethod.ORIGINAL:
        umbral = params.get('umbral', TARGET_THRESHOLD)
        next_returns = df_copy['close'].pct_change().shift(-1)
        df_copy['target'] = next_returns.apply(
            lambda x: 2 if x > umbral else (0 if x < -umbral else 1)
        )
        df_copy['target'] = df_copy['target'].where(next_returns.notna())
        
    elif method == TargetMethod.ATR:
        atr_period = params.get('atr_period', 14)
        atr_multiplier = params.get('atr_multiplier', 0.5)
        # Use ATR from our new indicators module
        df_copy = prepare_features(df_copy, include_all=False, 
                                 custom_config={'volatility': {'atr_window': atr_period}})
        threshold = df_copy['atr'] * atr_multiplier
        next_return = df_copy['close'].pct_change().shift(-1)
        df_copy['target'] = np.where(
            next_return > threshold, 2,
            np.where(next_return < -threshold, 0, 1)
        )
        
    elif method == TargetMethod.HORIZONTE_N:
        n_periods = params.get('n_periods', 5)
        umbral = params.get('umbral', TARGET_THRESHOLD)
        future_return = df_copy['close'].pct_change(periods=n_periods).shift(-n_periods)
        df_copy['target'] = future_return.apply(
            lambda x: 2 if x > umbral else (0 if x < -umbral else 1)
        )
        df_copy['target'] = df_copy['target'].where(future_return.notna())
        
    elif method == TargetMethod.NIVEL_ALCANZADO:
        n_periods = params.get('n_periods', 5)
        umbral_ratio = params.get('umbral_ratio', TARGET_THRESHOLD * 1.5)
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
        mask_valid = df_copy['future_max_high'].notna() & df_copy['future_min_low'].notna()
        conditions = [df_copy['hit_up'], df_copy['hit_down']]
        choices = [2, 0]
        df_copy['target'] = np.where(mask_valid, np.select(conditions, choices, default=1), np.nan)
        
    return ddef analyze_feature_importance(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    save_dir: str = MODEL_SAVE_DIR
) -> Tuple[pd.DataFrame, None]:
    """
    Analiza la importancia de las características usando tanto la importancia del modelo
    como valores SHAP.

    Args:
        model: Modelo entrenado
        X: DataFrame con las características
        feature_names: Lista de nombres de características
        save_dir: Directorio donde guardar los gráficos

    Returns:
        DataFrame con las importancias de características
    """
    logging.info("Analizando importancia de características...")
    
    try:
        # Obtener importancia basada en el modelo
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            logging.warning("El modelo no proporciona importancia de características directamente")
            importance = np.zeros(len(feature_names))

        # Calcular valores SHAP
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_importance = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
            else:
                shap_importance = np.abs(shap_values).mean(axis=0)
        except Exception as e:
            logging.warning(f"No se pudieron calcular valores SHAP: {e}")
            shap_importance = np.zeros(len(feature_names))

        # Crear DataFrame de importancia
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'model_importance': importance,
            'shap_importance': shap_importance
        })

        # Calcular importancia combinada
        importance_df['combined_importance'] = (
            (importance_df['model_importance'] / importance_df['model_importance'].max() if importance_df['model_importance'].max() > 0 else 0) +
            (importance_df['shap_importance'] / importance_df['shap_importance'].max() if importance_df['shap_importance'].max() > 0 else 0)
        ) / 2

        importance_df = importance_df.sort_values('combined_importance', ascending=False)

        # Crear visualizaciones
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        importance_df.head(15).plot(x='feature', y='model_importance', kind='bar')
        plt.title('Importancia de Características (Modelo)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        importance_df.head(15).plot(x='feature', y='shap_importance', kind='bar')
        plt.title('Importancia de Características (SHAP)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Guardar gráfico
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"{save_dir}/feature_importance_{timestamp}.png", bbox_inches='tight')
        plt.close()

        return importance_df, None

    except Exception as e:
        logging.error(f"Error en el análisis de importancia de características: {e}")
        return pd.DataFrame(), None


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
    y permitiendo diferentes métodos para definir el target.

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
    df = definir_target(df, target_method, **target_params)
    if df is None or 'target' not in df.columns:
        logging.error(f"El método target '{target_method}' no generó una columna 'target' válida.")
        return None
    # --- 2. Ingeniería de Características Avanzada ---
    logging.info("Calculando características avanzadas...")
    try:
        # Usar nuestro nuevo módulo de indicadores
        df = prepare_features(df, include_all=True)
        
        # Asegurarse de que tenemos las características básicas
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
            
        # Lista actualizada de características
        features_cols = [
            'open', 'high', 'low', 'close', 'temporalidad', 'returns',
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_high', 'bb_mid', 'bb_low', 'bb_width',
            'stoch_k', 'stoch_d',
            'adx', 'di_plus', 'di_minus',
            'atr', 'natr',
            'obv', 'volume_sma', 'volume_ratio',
            'price_std', 'volatility_ratio'
        ]
        
        # Filtrar solo las columnas que existen
        features_cols = [col for col in features_cols if col in df.columns]
        
    except Exception as e:
        logging.error(f"Error en el cálculo de características: {e}")
        return None
    # --- 3. Limpieza Final de Datos ---
    initial_rows: int = len(df)
    df = df.dropna(subset=['target']) # Asegura que el target calculado sea válido
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

        # Analizar importancia de características
        if hasattr(grid_search.best_estimator_, 'feature_importances_') or hasattr(grid_search.best_estimator_, 'coef_'):
            importance_df, _ = analyze_feature_importance(
                grid_search.best_estimator_,
                X_train,
                features_cols
            )
            logging.info("\nImportancia de características top 10:")
            logging.info(importance_df.head(10).to_string())
        else:
            logging.warning("El modelo no proporciona información de importancia de características")
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
    # Comprobar si fit_params tiene contenido antes de pasarlo
    if fit_params:
         cv_params = fit_params
    else:
         cv_params = None # Pasar None si está vacío

    try:
        # scores es ndarray
        scores: np.ndarray = cross_val_score(best_model, X_train, y_train, cv=tscv, scoring='f1_weighted',
                                            params=cv_params, n_jobs=-1) # type: ignore[arg-type] # Ignorar si Pylance se queja de params=None
        logging.info(f"Cross-Validation F1-Weighted Scores (en entreno): {scores}")
        logging.info(f"Media CV F1 (entreno): {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    except Exception as e:
        logging.warning(f"No se pudo calcular cross_val_score opcional: {e}", exc_info=True)


    logging.info('✅ Entrenamiento de modelo finalizado.')
    return best_model