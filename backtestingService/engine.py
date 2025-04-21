# backtestingService/engine.py

import logging
import pandas as pd
import numpy as np
import itertools
import os
from typing import List, Tuple, Any, Optional, Dict, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pandas import Timestamp
# Import our custom indicators module
from utils.indicators import prepare_features
from utils.trade_utils import calcular_cantidad

# Import configuration
from config.config import (
    INITIAL_CAPITAL, RISK_PERCENT, RR_RATIO, 
    COMMISSION_PER_TRADE, SIMULATED_SPREAD,
    CONFIRMATION_PARAMS, MIN_POSITION_SIZE, MAX_POSITION_SIZE,
    POSITION_PARAMS
)
from config.config import get_confirmation_parameters, get_risk_parameters
import binanceService.api as binance_api
# Definir un tipo más específico para la estructura de un trade para claridad
# (Tipo, Precio Entrada, Nivel Salida, Fecha Entrada, Fecha Salida)
TradeTuple = Tuple[str, float, float, Timestamp, Timestamp]

@dataclass
class BacktestStats:
    """Statistics from backtest execution."""
    total_trades: int
    winning_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_return: float
    sharpe_ratio: float
    avg_trade_duration: timedelta
    max_consecutive_losses: int


def calculate_backtest_stats(
    trades: List[TradeTuple],
    balance_hist: List[float],
    initial_capital: float
) -> BacktestStats:
    """
    Calculate comprehensive statistics from backtest results.
    
    Args:
        trades: List of executed trades
        balance_hist: History of account balance
        initial_capital: Starting capital
        
    Returns:
        BacktestStats object with calculated metrics
    """
    if not trades:
        return BacktestStats(
            total_trades=0,
            winning_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            total_return=0.0,
            sharpe_ratio=0.0,
            avg_trade_duration=timedelta(0),
            max_consecutive_losses=0
        )

    # Basic statistics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t[0].endswith('GANANCIA')])
    win_rate = (winning_trades / total_trades) * 100

    # Profit metrics
    returns = np.diff(balance_hist) / balance_hist[:-1]
    total_return = ((balance_hist[-1] - initial_capital) / initial_capital) * 100
    
    # Drawdown calculation
    cummax = np.maximum.accumulate(balance_hist)
    drawdown = (cummax - balance_hist) / cummax
    max_drawdown = np.max(drawdown) * 100

    # Risk metrics
    if len(returns) > 1:
        sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns))
    else:
        sharpe_ratio = 0.0

    # Trade duration
    durations = [(t[4] - t[3]) for t in trades]
    avg_duration = sum(durations, timedelta(0)) / len(durations) if durations else timedelta(0)

    # Consecutive losses
    results = [1 if t[0].endswith('GANANCIA') else 0 for t in trades]
    max_consecutive_losses = max(
        len(list(group))
        for key, group in itertools.groupby(results)
        if key == 0
    ) if results else 0

    # Calculate profit factor
    profits = sum(
        float(t[2] - t[1])
        for t in trades
        if t[0].endswith('GANANCIA')
    )
    losses = abs(sum(
        float(t[2] - t[1])
        for t in trades
        if t[0].endswith('PERDIDA')
    ))
    profit_factor = profits / losses if losses > 0 else float('inf')

    return BacktestStats(
        total_trades=total_trades,
        winning_trades=winning_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        avg_trade_duration=avg_duration,
        max_consecutive_losses=max_consecutive_losses
    )


# Backtesting y simulación de resultados financieros
def realizar_backtest(df: pd.DataFrame, model: Any) -> Tuple[float, List[TradeTuple], List[float]]:
    """
    Realiza un backtest avanzado con gestión dinámica del riesgo y tamaño de posición.
    
    Args:
        df: DataFrame de pandas con datos OHLCV y columnas de características requeridas.
            Se espera que el índice sea de tipo Timestamp.
        model: Modelo de ML entrenado con un método predict(features).
               Se usa Any ya que el tipo exacto puede variar (RandomForest, XGBoost, etc.).

    Returns:
        Una tupla conteniendo:
        - balance final (float)
        - lista de trades ejecutados (List[TradeTuple])
        - historial del balance (List[float])
    """
    logging.info("Iniciando backtest con gestión dinámica de riesgo...")
    
    # Preparar datos y características
    df_processed = prepare_features(df.copy(), include_all=True)
    
    if df_processed.empty:
        logging.warning("DataFrame vacío después de preparar features. No se puede realizar backtest.")
        return INITIAL_CAPITAL, [], [INITIAL_CAPITAL]
    
    # Obtener parámetros
    risk_params = get_risk_parameters()
    balance: float = INITIAL_CAPITAL
    balance_hist: List[float] = [balance]
    trades: List[TradeTuple] = []
    
    # Definir características requeridas
    feature_cols = [
        'open', 'high', 'low', 'close', 'returns',
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_high', 'bb_mid', 'bb_low', 'bb_width',
        'stoch_k', 'stoch_d',
        'adx', 'di_plus', 'di_minus',
        'atr', 'volume_ratio',
        'price_std', 'volatility_ratio'
    ]
    
    # Filtrar características existentes
    existing_features = [col for col in feature_cols if col in df_processed.columns]
    
    if len(existing_features) < 10:
        logging.warning(f"Faltan demasiadas características: solo {len(existing_features)} de {len(feature_cols)} disponibles")
        return INITIAL_CAPITAL, [], [INITIAL_CAPITAL]
    
    # Iterar sobre los datos
    for i in range(len(df_processed) - 1):
        try:
            # Obtener datos actuales
            current_data = df_processed.iloc[i:i+1]
            if current_data[existing_features].isna().any().any():
                logging.debug(f"Datos incompletos en índice {i}. Saltando iteración.")
                balance_hist.append(balance)
                continue
            
            # Obtener predicción
            features = current_data[existing_features]
            prediccion = model.predict(features)[0]
            
            # Obtener condiciones de mercado actuales
            current_price = float(current_data['close'].iloc[0])
            current_atr = float(current_data['atr'].iloc[0])
            current_adx = float(current_data['adx'].iloc[0])
            current_volume_ratio = float(current_data['volume_ratio'].iloc[0])
            fecha_entrada = df_processed.index[i]
            
            # Confirmar señal
            df_subset = df_processed.iloc[:i+1]
            confirmar_long, confirmar_short = confirmar_senal_con_indicadores(df_subset)
            
            # Calcular tamaño de posición
            position_size = calculate_position_size(
                balance,
                current_price,
                current_atr,
                current_adx,
                current_volume_ratio,
                risk_params
            )
            
            trade_executed = False
            # Procesar señal LONG
            if prediccion == 2 and confirmar_long:
                # Check portfolio risk before executing trade
                if not check_portfolio_risk(
                    current_trades=[t for t in trades if t[0].startswith(('LONG', 'SHORT'))],
                    new_trade_direction='LONG',
                    new_trade_size=position_size,
                    df=df_processed
                ):
                    logging.info("Portfolio risk check failed for LONG trade")
                    continue
                
                precio_entrada = current_price * (1 + SIMULATED_SPREAD)
                initial_stop_loss, final_take_profit = calculate_stop_levels(
                    df_subset, 'LONG', precio_entrada, current_atr, risk_params
                )
                
                # Calculate partial take-profit levels
                tp_levels = calculate_partial_tp_levels(precio_entrada, initial_stop_loss, 'LONG')
                
                # Initialize position tracking
                remaining_position = position_size
                current_stop = initial_stop_loss
                risk_amount = position_size * (precio_entrada - initial_stop_loss) / precio_entrada
                
                for j in range(i + 1, len(df_processed)):
                    fecha_salida = df_processed.index[j]
                    low_price = df_processed['low'].iloc[j]
                    high_price = df_processed['high'].iloc[j]
                    close_price = df_processed['close'].iloc[j]
                    current_atr = float(df_processed['atr'].iloc[j])
                    
                    # Update trailing stop
                    current_stop = update_trailing_stop(
                        close_price, initial_stop_loss, current_stop, 'LONG', current_atr
                    )
                    
                    # Check stop loss
                    if low_price <= current_stop:
                        loss = remaining_position * (precio_entrada - current_stop) / precio_entrada
                        balance -= abs(loss)
                        balance -= (remaining_position * COMMISSION_PER_TRADE * 2)
                        trades.append(('LONG_PERDIDA', precio_entrada, current_stop, fecha_entrada, fecha_salida))
                        trade_executed = True
                        break
                    
                    # Check partial take-profits
                    for tp_price, tp_percent in list(tp_levels):  # Create a copy for safe iteration
                        if high_price >= tp_price and remaining_position > 0:
                            partial_size = position_size * tp_percent
                            partial_profit = partial_size * (tp_price - precio_entrada) / precio_entrada
                            balance += partial_profit
                            balance -= (partial_size * COMMISSION_PER_TRADE * 2)
                            remaining_position -= partial_size
                            trades.append(('LONG_TP_PARCIAL', precio_entrada, tp_price, fecha_entrada, fecha_salida))
                            
                            # Remove this level from future checks
                            tp_levels = [(p, s) for p, s in tp_levels if p > tp_price]
                    
                    # Check final take-profit for remaining position
                    if remaining_position > 0 and high_price >= final_take_profit:
                        final_profit = remaining_position * (final_take_profit - precio_entrada) / precio_entrada
                        balance += final_profit
                        balance -= (remaining_position * COMMISSION_PER_TRADE * 2)
                        trades.append(('LONG_GANANCIA', precio_entrada, final_take_profit, fecha_entrada, fecha_salida))
                        trade_executed = True
                        break
            elif prediccion == 0 and confirmar_short:
                # Check portfolio risk before executing trade
                if not check_portfolio_risk(
                    current_trades=[t for t in trades if t[0].startswith(('LONG', 'SHORT'))],
                    new_trade_direction='SHORT',
                    new_trade_size=position_size,
                    df=df_processed
                ):
                    logging.info("Portfolio risk check failed for SHORT trade")
                    continue
                    
                precio_entrada = current_price * (1 - SIMULATED_SPREAD)
                initial_stop_loss, final_take_profit = calculate_stop_levels(
                    df_subset, 'SHORT', precio_entrada, current_atr, risk_params
                )
                
                # Calculate partial take-profit levels
                tp_levels = calculate_partial_tp_levels(precio_entrada, initial_stop_loss, 'SHORT')
                
                # Initialize position tracking
                remaining_position = position_size
                current_stop = initial_stop_loss
                risk_amount = position_size * (current_stop - precio_entrada) / precio_entrada
                
                for j in range(i + 1, len(df_processed)):
                    fecha_salida = df_processed.index[j]
                    low_price = df_processed['low'].iloc[j]
                    high_price = df_processed['high'].iloc[j]
                    close_price = df_processed['close'].iloc[j]
                    current_atr = float(df_processed['atr'].iloc[j])
                    
                    # Update trailing stop
                    current_stop = update_trailing_stop(
                        close_price, initial_stop_loss, current_stop, 'SHORT', current_atr
                    )
                    
                    # Check stop loss
                    if high_price >= current_stop:
                        loss = remaining_position * (current_stop - precio_entrada) / precio_entrada
                        balance -= abs(loss)
                        balance -= (remaining_position * COMMISSION_PER_TRADE * 2)
                        trades.append(('SHORT_PERDIDA', precio_entrada, current_stop, fecha_entrada, fecha_salida))
                        trade_executed = True
                        break
                    
                    # Check partial take-profits
                    for tp_price, tp_percent in list(tp_levels):  # Create a copy for safe iteration
                        if low_price <= tp_price and remaining_position > 0:
                            partial_size = position_size * tp_percent
                            partial_profit = partial_size * (precio_entrada - tp_price) / precio_entrada
                            balance += partial_profit
                            balance -= (partial_size * COMMISSION_PER_TRADE * 2)
                            remaining_position -= partial_size
                            trades.append(('SHORT_TP_PARCIAL', precio_entrada, tp_price, fecha_entrada, fecha_salida))
                            
                            # Remove this level from future checks
                            tp_levels = [(p, s) for p, s in tp_levels if p < tp_price]
                    
                    # Check final take-profit for remaining position
                    if remaining_position > 0 and low_price <= final_take_profit:
                        final_profit = remaining_position * (precio_entrada - final_take_profit) / precio_entrada
                        balance += final_profit
                        balance -= (remaining_position * COMMISSION_PER_TRADE * 2)
                        trades.append(('SHORT_GANANCIA', precio_entrada, final_take_profit, fecha_entrada, fecha_salida))
                        trade_executed = True
                        break
            # Registrar balance
            balance_hist.append(balance)
            
            # Logging detallado para trades ejecutados
            if trade_executed and logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    f"Trade executed at {fecha_entrada}:\n"
                    f"Type: {'LONG' if prediccion == 2 else 'SHORT'}\n"
                    f"Entry Price: {precio_entrada:.4f}\n"
                    f"Stop Loss: {stop_loss:.4f}\n"
                    f"Take Profit: {take_profit:.4f}\n"
                    f"Position Size: {position_size:.2f}\n"
                    f"Risk Amount: {risk_amount:.2f}\n"
                    f"Balance After Trade: {balance:.2f}"
                )
            
            # Avanzar el índice después de un trade para evitar señales muy cercanas
            if trade_executed:
                i = j  # Saltar hasta después del cierre del trade
                
        except Exception as e:
            logging.error(f"Error en iteración {i}: {e}", exc_info=True)
            balance_hist.append(balance)
            continue
    
    # Calcular y loggear estadísticas finales
    if trades:
        stats = calculate_backtest_stats(trades, balance_hist, INITIAL_CAPITAL)
        logging.info(
            f"\nBacktesting Statistics:\n"
            f"Total Trades: {stats.total_trades}\n"
            f"Win Rate: {stats.win_rate:.2f}%\n"
            f"Profit Factor: {stats.profit_factor:.2f}\n"
            f"Max Drawdown: {stats.max_drawdown:.2f}%\n"
            f"Sharpe Ratio: {stats.sharpe_ratio:.2f}\n"
            f"Average Trade Duration: {stats.avg_trade_duration}\n"
            f"Max Consecutive Losses: {stats.max_consecutive_losses}\n"
            f"Final Balance: {balance:.2f}\n"
            f"Total Return: {stats.total_return:.2f}%"
        )
        
        # Generate backtest report with visualizations
        generate_backtest_report(df_processed, trades, balance_hist, stats, BACKTEST_DIR)
        
    return balance, trades, balance_hist

# Función para confirmar la señal con indicadores técnicos
def confirmar_senal_con_indicadores(df: pd.DataFrame) -> Tuple[bool, bool]:
    """
    Confirms trading signals using multiple technical indicators and market conditions.

    Args:
        df: DataFrame with OHLCV data and indicators. Must have sufficient
            rows to calculate all required indicators.

    Returns:
        Tuple of (confirmar_long, confirmar_short) booleans.
    """
    # Get confirmation parameters
    conf_params = get_confirmation_parameters()
    
    # Work with a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    try:
        # Use our prepare_features function to calculate all indicators
        df_processed = prepare_features(df_copy, include_all=True)
        
        # Check if we have enough data
        required_indicators = [
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_high', 'bb_mid', 'bb_low', 'bb_width',
            'stoch_k', 'stoch_d',
            'adx', 'di_plus', 'di_minus',
            'atr', 'volume_ratio',
            'price_std', 'volatility_ratio'
        ]
        
        # Filter to only include indicators that exist in the DataFrame
        existing_indicators = [ind for ind in required_indicators if ind in df_processed.columns]
        
        if df_processed.empty or len(existing_indicators) < len(required_indicators) / 2:
            logging.warning("Insufficient indicators for signal confirmation. Need to calculate more indicators.")
            return False, False
            
        if df_processed[existing_indicators].iloc[-1].isna().any():
            logging.warning("NaN values in key indicators for signal confirmation.")
            return False, False
        
        # Get latest values
        latest = df_processed.iloc[-1]
        
        # --- Trend Analysis ---
        trend_strength = latest['adx'] > conf_params['MIN_ADX']
        strong_trend_up = (
            latest['di_plus'] > latest['di_minus'] and
            latest['macd'] > conf_params['MACD_THRESHOLD'] and
            latest['macd_diff'] > 0
        )
        strong_trend_down = (
            latest['di_minus'] > latest['di_plus'] and
            latest['macd'] < -conf_params['MACD_THRESHOLD'] and
            latest['macd_diff'] < 0
        )
        
        # --- Momentum Analysis ---
        momentum_bull = (
            latest['rsi'] > 50 and
            latest['rsi'] < conf_params['RSI_OVERBOUGHT'] and
            latest['stoch_k'] > latest['stoch_d'] and
            latest['stoch_k'] < 80
        )
        momentum_bear = (
            latest['rsi'] < 50 and
            latest['rsi'] > conf_params['RSI_OVERSOLD'] and
            latest['stoch_k'] < latest['stoch_d'] and
            latest['stoch_k'] > 20
        )
        
        # --- Volume Analysis ---
        volume_confirmation = latest['volume_ratio'] > conf_params['MIN_VOLUME_RATIO']
        
        # --- Volatility Analysis ---
        volatility_reasonable = (
            latest['bb_width'] > conf_params['BB_THRESHOLD'] and
            latest['volatility_ratio'] < 2.0  # Not excessive volatility
        )
        
        # --- Price Structure Analysis ---
        # Check if price is in favorable position relative to Bollinger Bands
        price_above_mid = latest['close'] > latest['bb_mid']
        price_below_mid = latest['close'] < latest['bb_mid']
        not_overbought = latest['close'] < latest['bb_high'] * 0.95  # Not too close to upper band
        not_oversold = latest['close'] > latest['bb_low'] * 1.05   # Not too close to lower band
        
        # --- Combine All Conditions ---
        confirmar_long = (
            strong_trend_up and
            trend_strength and
            momentum_bull and
            volume_confirmation and
            volatility_reasonable and
            price_above_mid and
            not_overbought
        )
        
        confirmar_short = (
            strong_trend_down and
            trend_strength and
            momentum_bear and
            volume_confirmation and
            volatility_reasonable and
            price_below_mid and
            not_oversold
        )
        
        # Log confirmation details if debug logging is enabled
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                f"Signal Confirmation Details:\n"
                f"Trend Strength: {trend_strength}\n"
                f"Strong Trend Up/Down: {strong_trend_up}/{strong_trend_down}\n"
                f"Momentum Bull/Bear: {momentum_bull}/{momentum_bear}\n"
                f"Volume Confirmation: {volume_confirmation}\n"
                f"Volatility Reasonable: {volatility_reasonable}\n"
                f"Price Structure (Above/Below Mid): {price_above_mid}/{price_below_mid}\n"
                f"Final Confirmation Long/Short: {confirmar_long}/{confirmar_short}"
            )
        
        return confirmar_long, confirmar_short
        
    except Exception as e:
        logging.error(f"Error in signal confirmation: {e}", exc_info=True)
        return False, False


def check_portfolio_risk(
    current_trades: List[TradeTuple],
    new_trade_direction: str,
    new_trade_size: float,
    df: pd.DataFrame
) -> bool:
    """
    Check if adding a new trade maintains acceptable portfolio risk levels.
    
    Args:
        current_trades: List of currently open trades
        new_trade_direction: Direction of the new trade ('LONG' or 'SHORT')
        new_trade_size: Size of the new position
        df: DataFrame with price data
        
    Returns:
        Boolean indicating if new trade is acceptable
    """
    try:
        # Check maximum number of positions
        if len(current_trades) >= POSITION_PARAMS['MAX_POSITIONS']:
            logging.warning(f"Maximum number of positions ({POSITION_PARAMS['MAX_POSITIONS']}) reached")
            return False
            
        # Calculate current portfolio exposure
        current_exposure = sum(abs(
            float(trade[1] - trade[2])  # Entry price - Exit level
            for trade in current_trades
            if not trade[0].endswith(('GANANCIA', 'PERDIDA'))  # Only count open trades
        ))
        
        # Check if new trade would exceed maximum portfolio exposure
        max_exposure = INITIAL_CAPITAL * MAX_POSITION_SIZE
        if current_exposure + new_trade_size > max_exposure:
            logging.warning(
                f"New trade would exceed maximum portfolio exposure "
                f"({(current_exposure + new_trade_size):.2f} > {max_exposure:.2f})"
            )
            return False
        
        # Check correlation with existing positions
        if current_trades and len(df) > 20:  # Need enough data for correlation
            # Calculate returns
            returns = df['close'].pct_change()
            
            # Calculate correlation with existing trades
            for trade in current_trades:
                if not trade[0].endswith(('GANANCIA', 'PERDIDA')):  # Only check open trades
                    # Get trade direction
                    trade_direction = 1 if trade[0].startswith('LONG') else -1
                    new_direction = 1 if new_trade_direction == 'LONG' else -1
                    
                    # Calculate correlation of returns
                    trade_period = df.index >= trade[3]  # From trade entry
                    if trade_period.any():
                        trade_returns = returns[trade_period] * trade_direction
                        new_returns = returns[trade_period] * new_direction
                        
                        correlation = trade_returns.corr(new_returns)
                        if abs(correlation) > POSITION_PARAMS['MAX_CORRELATION']:
                            logging.warning(
                                f"Trade correlation too high: {correlation:.2f} > "
                                f"{POSITION_PARAMS['MAX_CORRELATION']}"
                            )
                            return False
        
        # Calculate portfolio heat (total risk relative to capital)
        portfolio_heat = sum(
            abs(float(trade[2] - trade[1])) / float(trade[1])  # (SL - Entry) / Entry
            for trade in current_trades
            if not trade[0].endswith(('GANANCIA', 'PERDIDA'))
        )
        
        # Check if new trade would exceed maximum portfolio heat
        max_heat = 0.1  # Maximum 10% total portfolio risk
        if portfolio_heat > max_heat:
            logging.warning(
                f"Portfolio heat too high: {portfolio_heat:.2%} > {max_heat:.2%}"
            )
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error checking portfolio risk: {e}", exc_info=True)
        return False


# Funciones para cálculo dinámico de posición y niveles de stop
def calculate_position_size(
    balance: float,
    price: float,
    atr: float,
    adx: float,
    volume_ratio: float,
    params: Dict[str, float]
) -> float:
    """
    Calculate position size dynamically based on market conditions.
    
    Args:
        balance: Current account balance
        price: Current price
        atr: Average True Range value
        adx: ADX value for trend strength
        volume_ratio: Current volume relative to average
        params: Risk parameters from config
        
    Returns:
        Position size in quote currency
    """
    # Base risk percent adjusted by ATR
    normalized_atr = atr / price
    risk_percent = np.clip(
        params['base_risk_percent'] * (1 - normalized_atr * params['atr_multiple']),
        params['min_risk_percent'],
        params['max_risk_percent']
    )
    
    # Adjust for trend strength
    if adx > CONFIRMATION_PARAMS['MIN_ADX']:
        risk_percent *= (1 + (adx - CONFIRMATION_PARAMS['MIN_ADX']) / 75 * params['trend_strength_factor'])
    
    # Adjust for volume
    if volume_ratio > CONFIRMATION_PARAMS['MIN_VOLUME_RATIO']:
        risk_percent *= (1 + (volume_ratio - 1) * params['volume_factor'])
    
    position_size = balance * risk_percent
    return np.clip(position_size, balance * MIN_POSITION_SIZE, balance * MAX_POSITION_SIZE)

def calculate_stop_levels(
    df: pd.DataFrame,
    direction: str,
    current_price: float,
    atr: float,
    params: Dict[str, float]
) -> Tuple[float, float]:
    """
    Calculate dynamic stop loss and take profit levels.
    
    Args:
        df: DataFrame with price data
        direction: 'LONG' or 'SHORT'
        current_price: Entry price
        atr: Current ATR value
        params: Risk parameters
        
    Returns:
        Tuple of (stop_loss, take_profit) levels
    """
    atr_multiple = params['atr_multiple']
    rr_ratio = params['rr_ratio']
    
    if direction == 'LONG':
        recent_low = df['low'].tail(5).min()
        sl_technical = min(recent_low, current_price - atr * atr_multiple)
        sl_distance = current_price - sl_technical
        stop_loss = current_price - sl_distance
        take_profit = current_price + (sl_distance * rr_ratio)
    else:  # SHORT
        recent_high = df['high'].tail(5).max()
        sl_technical = max(recent_high, current_price + atr * atr_multiple)
        sl_distance = sl_technical - current_price
        stop_loss = current_price + sl_distance
        take_profit = current_price - (sl_distance * rr_ratio)
    
    return stop_loss, take_profit

def calculate_partial_tp_levels(
    entry_price: float,
    stop_loss: float,
    direction: str
) -> List[Tuple[float, float]]:
    """
    Calculate partial take-profit levels based on risk multiples.
    
    Args:
        entry_price: Trade entry price
        stop_loss: Stop loss price
        direction: 'LONG' or 'SHORT'
        
    Returns:
        List of tuples (take_profit_price, position_percent)
    """
    if direction == 'LONG':
        risk = entry_price - stop_loss
        tp_levels = [
            (entry_price + risk * POSITION_PARAMS['PARTIAL_TAKE_PROFIT']['LEVEL_1']['at_price'],
             POSITION_PARAMS['PARTIAL_TAKE_PROFIT']['LEVEL_1']['percent']),
            (entry_price + risk * POSITION_PARAMS['PARTIAL_TAKE_PROFIT']['LEVEL_2']['at_price'],
             POSITION_PARAMS['PARTIAL_TAKE_PROFIT']['LEVEL_2']['percent'])
        ]
    else:  # SHORT
        risk = stop_loss - entry_price
        tp_levels = [
            (entry_price - risk * POSITION_PARAMS['PARTIAL_TAKE_PROFIT']['LEVEL_1']['at_price'],
             POSITION_PARAMS['PARTIAL_TAKE_PROFIT']['LEVEL_1']['percent']),
            (entry_price - risk * POSITION_PARAMS['PARTIAL_TAKE_PROFIT']['LEVEL_2']['at_price'],
             POSITION_PARAMS['PARTIAL_TAKE_PROFIT']['LEVEL_2']['percent'])
        ]
    
    return tp_levels

def update_trailing_stop(
    current_price: float,
    initial_stop: float,
    current_stop: float,
    direction: str,
    atr: float
) -> float:
    """
    Update trailing stop based on price movement and ATR.
    
    Args:
        current_price: Current market price
        initial_stop: Initial stop loss level
        current_stop: Current stop loss level
        direction: 'LONG' or 'SHORT'
        atr: Current ATR value
        
    Returns:
        Updated stop loss level
    """
    if direction == 'LONG':
        # Calculate potential new stop based on ATR
        potential_stop = current_price - (atr * POSITION_PARAMS['TRAILING_STOP']['STEP'])
        
        # Only update if it would move the stop higher
        if potential_stop > current_stop and current_price > initial_stop * POSITION_PARAMS['TRAILING_STOP']['ACTIVATION']:
            return max(potential_stop, current_stop)
    else:  # SHORT
        potential_stop = current_price + (atr * POSITION_PARAMS['TRAILING_STOP']['STEP'])
        if potential_stop < current_stop and current_price < initial_stop * POSITION_PARAMS['TRAILING_STOP']['ACTIVATION']:
            return min(potential_stop, current_stop)
    
    return current_stop
# La función calcular_cantidad se ha movido a utils/trade_utils.py para evitar importaciones circulares


def generate_backtest_report(
    df: pd.DataFrame,
    trades: List[TradeTuple],
    balance_hist: List[float],
    stats: BacktestStats,
    save_dir: str = None
) -> None:
    """
    Generate comprehensive backtest report with visualizations.
    
    Args:
        df: DataFrame with price data and indicators
        trades: List of executed trades
        balance_hist: History of account balance
        stats: Calculated backtest statistics
        save_dir: Optional directory to save report files
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'backtest_reports')
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # Set style
        plt.style.use('seaborn')
        
        # Create main figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Price chart with trades
        ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2, colspan=2)
        ax1.plot(df.index, df['close'], label='Price', color='blue', alpha=0.7)
        
        # Plot trades
        for trade in trades:
            trade_type, entry_price, exit_price, entry_time, exit_time = trade
            
            # Define colors and markers based on trade type
            if trade_type.startswith('LONG'):
                color = 'green' if trade_type.endswith('GANANCIA') else 'red'
                marker = '^'
            else:  # SHORT
                color = 'green' if trade_type.endswith('GANANCIA') else 'red'
                marker = 'v'
            
            # Plot entry and exit points
            ax1.scatter(entry_time, entry_price, color=color, marker=marker, s=100)
            if trade_type.endswith(('GANANCIA', 'PERDIDA')):
                ax1.scatter(exit_time, exit_price, color='black', marker='x', s=100)
                ax1.plot([entry_time, exit_time], [entry_price, exit_price], 
                        color=color, linestyle='--', alpha=0.5)
        
        ax1.set_title('Price Chart with Trades')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # 2. Equity curve
        ax2 = plt.subplot2grid((3, 2), (2, 0))
        equity_line = ax2.plot(df.index[:len(balance_hist)], balance_hist, 
                             label='Equity', color='green')
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Balance')
        ax2.grid(True)
        
        # Add drawdown shading
        cummax = np.maximum.accumulate(balance_hist)
        drawdown = (cummax - balance_hist) / cummax * 100
        ax2.fill_between(df.index[:len(drawdown)], 0, drawdown, 
                        alpha=0.3, color='red', label='Drawdown')
        ax2.legend()
        
        # 3. Trade duration histogram
        ax3 = plt.subplot2grid((3, 2), (2, 1))
        durations = [(t[4] - t[3]).total_seconds() / 3600 for t in trades]  # Convert to hours
        ax3.hist(durations, bins=20, edgecolor='black')
        ax3.set_title('Trade Duration Distribution (Hours)')
        ax3.set_xlabel('Duration (hours)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'backtest_overview_{timestamp}.png'))
        plt.close()
        
        # Additional analysis plots
        # 1. Monthly returns heatmap
        returns = pd.Series(np.diff(balance_hist) / balance_hist[:-1], 
                          index=df.index[1:len(balance_hist)])
        monthly_returns = returns.resample('M').sum().unstack()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn')
        plt.title('Monthly Returns Heatmap')
        plt.savefig(os.path.join(save_dir, f'monthly_returns_{timestamp}.png'))
        plt.close()
        
        # 2. Trade analysis by hour
        trade_data = pd.DataFrame([
            {
                'entry_time': t[3],
                'result': 'Win' if t[0].endswith('GANANCIA') else 'Loss',
                'type': 'Long' if t[0].startswith('LONG') else 'Short',
                'return': (t[2] - t[1]) / t[1] * (1 if t[0].startswith('LONG') else -1)
            }
            for t in trades
        ])
        
        if not trade_data.empty:
            trade_data['hour'] = trade_data['entry_time'].dt.hour
            
            # Win rate by hour
            plt.figure(figsize=(12, 6))
            hourly_winrate = trade_data.pivot_table(
                index='hour',
                values='result',
                aggfunc=lambda x: (x == 'Win').mean()
            )
            hourly_winrate.plot(kind='bar')
            plt.title('Win Rate by Hour')
            plt.xlabel('Hour')
            plt.ylabel('Win Rate')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'hourly_winrate_{timestamp}.png'))
            plt.close()
        
        # Generate detailed report
        report_path = os.path.join(save_dir, f'backtest_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write("=== Backtest Performance Report ===\n\n")
            
            # Overall statistics
            f.write("--- Overall Performance ---\n")
            f.write(f"Total Trades: {stats.total_trades}\n")
            f.write(f"Winning Trades: {stats.winning_trades}\n")
            f.write(f"Win Rate: {stats.win_rate:.2f}%\n")
            f.write(f"Profit Factor: {stats.profit_factor:.2f}\n")
            f.write(f"Maximum Drawdown: {stats.max_drawdown:.2f}%\n")
            f.write(f"Sharpe Ratio: {stats.sharpe_ratio:.2f}\n")
            f.write(f"Average Trade Duration: {stats.avg_trade_duration}\n")
            f.write(f"Max Consecutive Losses: {stats.max_consecutive_losses}\n")
            f.write(f"Total Return: {stats.total_return:.2f}%\n\n")
            
            # Trade analysis
            if not trade_data.empty:
                f.write("--- Trade Analysis ---\n")
                f.write("\nBy Direction:\n")
                direction_stats = trade_data.groupby('type').agg({
                    'result': lambda x: (x == 'Win').mean(),
                    'return': ['count', 'mean', 'std', 'sum']
                })
                f.write(f"{direction_stats}\n\n")
                
                f.write("\nBy Hour:\n")
                hour_stats = trade_data.groupby('hour').agg({
                    'result': lambda x: (x == 'Win').mean(),
                    'return': ['count', 'mean', 'sum']
                })
                f.write(f"{hour_stats}\n")
        
        logging.info(f"Backtest report generated in {save_dir}")
        
    except Exception as e:
        logging.error(f"Error generating backtest report: {e}", exc_info=True)


def monitor_live_performance(
    recent_trades: List[TradeTuple],
    evaluation_period: int = 30
) -> Tuple[bool, Dict[str, float]]:
    """
    Monitor live trading performance and check for degradation.
    
    Args:
        recent_trades: List of recent trades
        evaluation_period: Days to look back for performance evaluation
        
    Returns:
        Tuple of (performance_acceptable, metrics_dict)
    """
    from datetime import datetime, timedelta
    
    try:
        # Filter trades within evaluation period
        cutoff_date = datetime.now() - timedelta(days=evaluation_period)
        recent_trades = [
            trade for trade in recent_trades
            if trade[3] >= cutoff_date  # Using entry time
        ]
        
        if not recent_trades:
            logging.warning(f"No trades in the last {evaluation_period} days")
            return True, {}
        
        # Calculate key metrics
        total_trades = len(recent_trades)
        winning_trades = len([t for t in recent_trades if t[0].endswith('GANANCIA')])
        win_rate = (winning_trades / total_trades) * 100
        
        # Calculate returns
        returns = [
            (t[2] - t[1]) / t[1] * (1 if t[0].startswith('LONG') else -1)
            for t in recent_trades
        ]
        avg_return = np.mean(returns) if returns else 0
        
        # Calculate profit factor
        profits = sum(r for r in returns if r > 0)
        losses = abs(sum(r for r in returns if r < 0))
        profit_factor = profits / losses if losses > 0 else float('inf')
        
        # Calculate drawdown
        cumulative_returns = np.cumsum(returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (rolling_max - cumulative_returns)
        max_drawdown = max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Calculate consecutive losses
        results = [1 if r > 0 else 0 for r in returns]
        max_consecutive_losses = max(
            len(list(group))
            for key, group in itertools.groupby(results)
            if key == 0
        ) if results else 0
        
        # Check against thresholds from config
        metrics = {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown * 100,  # Convert to percentage
            'avg_return': avg_return * 100,  # Convert to percentage
            'max_consecutive_losses': max_consecutive_losses,
            'total_trades': total_trades
        }
        
        # Performance checks
        performance_acceptable = (
            win_rate >= MONITORING_PARAMS['MIN_WIN_RATE'] and
            profit_factor >= MONITORING_PARAMS['MIN_PROFIT_FACTOR'] and
            max_drawdown <= MONITORING_PARAMS['MAX_DRAWDOWN'] and
            max_consecutive_losses <= 5  # Additional safety check
        )
        
        # Log performance metrics
        logging.info(
            f"\nLive Performance Metrics (Last {evaluation_period} days):\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Profit Factor: {profit_factor:.2f}\n"
            f"Max Drawdown: {max_drawdown*100:.2f}%\n"
            f"Average Return: {avg_return*100:.2f}%\n"
            f"Max Consecutive Losses: {max_consecutive_losses}\n"
            f"Total Trades: {total_trades}"
        )
        
        if not performance_acceptable:
            logging.warning(
                "⚠️ Performance degradation detected! Consider reviewing and recalibrating the model."
            )
        
        return performance_acceptable, metrics
        
    except Exception as e:
        logging.error(f"Error monitoring live performance: {e}", exc_info=True)
        return False, {}
