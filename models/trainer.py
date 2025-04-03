import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

from config.config import TARGET_THRESHOLD, RANDOM_STATE


# Funcion para entrenar un modelo con los datos de backtesting
def entrenar_modelo(model, df):

    logging.info('Iniciando entrenamiento de modelo')

    df = definir_target(df)

    df['returns'] = df['close'].pct_change()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = MACD(df['close']).macd_diff()
    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    df = df.dropna()
    #df = df[df['target'].isin([0, 2])]

    features = df[['open', 'high', 'low', 'close', 'temporalidad', 'returns', 'rsi', 'macd', 'ema9', 'ema21', 'atr']].astype(float)
    target = df['target']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=RANDOM_STATE, stratify=target)

    # Calcular class weights
    clases = np.unique(target)
    pesos = compute_class_weight(class_weight='balanced', classes=clases, y=target)
    class_weights_dict = {clase: peso for clase, peso in zip(clases, pesos)}

    # Ajustar el grid dependiendo del modelo
    model_name = model.__class__.__name__
    if model_name == 'RandomForestClassifier':
        model.set_params(class_weight=class_weights_dict)
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7],
            'max_features': ['auto', 'sqrt']
        }
    elif model_name == 'GradientBoostingClassifier':
        # no admite class_weight, lo usamos en sample_weight
        sample_weight = y_train.map(class_weights_dict)
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    elif model_name == 'XGBClassifier':
        # XGBoost usa scale_pos_weight (peso de clase minoritaria)
        scale_pos_weight = class_weights_dict[0] / class_weights_dict[2]
        model.set_params(scale_pos_weight=scale_pos_weight)
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    else:
        raise ValueError(f"Modelo '{model_name}' no soportado en tuning automático")

    if model_name == 'GradientBoostingClassifier':
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    scores = cross_val_score(best_model, features, target, cv=5)
    logging.info(f"✅ Modelo entrenado correctamente. Cross-Validation Score: {scores.mean():.4f}")

    y_pred = best_model.predict(X_test)
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

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


# Reentrena el modelo ya entrenado utilizando los resultados obtenidos en el backtest (TP o SL).
def refinar_modelo_con_resultados(modelo, df_original, historial_trades):

    X = []
    y = []

    for operacion in historial_trades:
        resultado, entrada, salida, fecha_entrada, _ = operacion
        etiqueta = 1 if "GANANCIA" in resultado else 0

        try:
            fila = df_original.loc[fecha_entrada:fecha_entrada].copy()
            if fila.empty:
                continue

            fila['returns'] = fila['close'].pct_change()
            fila['rsi'] = RSIIndicator(fila['close'], window=14).rsi()
            fila['macd'] = MACD(fila['close']).macd_diff()
            fila['ema9'] = EMAIndicator(fila['close'], window=9).ema_indicator()
            fila['ema21'] = EMAIndicator(fila['close'], window=21).ema_indicator()
            fila['atr'] = AverageTrueRange(fila['high'], fila['low'], fila['close'], window=14).average_true_range()

            fila.dropna(inplace=True)

            if not fila.empty:
                features = fila[['open', 'high', 'low', 'close', 'temporalidad', 'returns',
                                 'rsi', 'macd', 'ema9', 'ema21', 'atr']].iloc[0].values
                X.append(features)
                y.append(etiqueta)

        except Exception as e:
            print(f"Error procesando fila {fecha_entrada}: {e}")
            continue

    if not X:
        print("⚠️ No se encontraron datos suficientes para refinar el modelo.")
        return modelo

    X = pd.DataFrame(X, columns=['open', 'high', 'low', 'close', 'temporalidad', 'returns',
                                 'rsi', 'macd', 'ema9', 'ema21', 'atr'])
    y = pd.Series(y)

    print(f"📊 Refinando modelo con {len(y)} ejemplos de retroalimentación...")
    modelo.fit(X, y)

    print("✅ Modelo refinado exitosamente con feedback del backtest.")
    return modelo



