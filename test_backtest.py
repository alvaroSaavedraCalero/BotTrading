#!/usr/bin/env python3
# test_backtest.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from backtestingService.engine import (
    realizar_backtest,
    confirmar_senal_con_indicadores,
    check_portfolio_risk,
    calculate_position_size,
    calculate_stop_levels,
    calculate_partial_tp_levels,
    update_trailing_stop
)
from utils.indicators import prepare_features

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def generate_test_data(days=100):
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    # Generate prices with some trend and volatility
    close = np.random.randn(days).cumsum() + 100
    high = close + np.abs(np.random.randn(days)) * 0.5
    low = close - np.abs(np.random.randn(days)) * 0.5
    open_price = close + np.random.randn(days) * 0.2
    volume = np.abs(np.random.randn(days) * 1000) + 1000
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df

def test_signal_confirmation():
    """Test signal confirmation logic."""
    logging.info("Testing signal confirmation...")
    try:
        df = generate_test_data()
        df_processed = prepare_features(df, include_all=True)
        long_signal, short_signal = confirmar_senal_con_indicadores(df_processed)
        logging.info(f"Signal confirmation test passed. Long: {long_signal}, Short: {short_signal}")
        return True
    except Exception as e:
        logging.error(f"Signal confirmation test failed: {e}")
        return False

def test_position_sizing():
    """Test position sizing calculations."""
    logging.info("Testing position sizing...")
    try:
        balance = 10000.0
        price = 100.0
        atr = 2.0
        adx = 30.0
        volume_ratio = 1.2
        params = {
            'base_risk_percent': 0.01,
            'min_risk_percent': 0.005,
            'max_risk_percent': 0.02,
            'atr_multiple': 1.5,
            'trend_strength_factor': 0.5,
            'volume_factor': 1.0
        }
        
        position_size = calculate_position_size(
            balance, price, atr, adx, volume_ratio, params
        )
        logging.info(f"Position sizing test passed. Size: {position_size}")
        return True
    except Exception as e:
        logging.error(f"Position sizing test failed: {e}")
        return False

def test_stop_levels():
    """Test stop loss and take profit calculations."""
    logging.info("Testing stop levels calculation...")
    try:
        df = generate_test_data(20)
        current_price = float(df['close'].iloc[-1])
        atr = float(df['close'].std())
        params = {
            'atr_multiple': 1.5,
            'rr_ratio': 2.0
        }
        
        stop_loss, take_profit = calculate_stop_levels(
            df, 'LONG', current_price, atr, params
        )
        logging.info(
            f"Stop levels test passed. Entry: {current_price:.2f}, "
            f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}"
        )
        return True
    except Exception as e:
        logging.error(f"Stop levels test failed: {e}")
        return False

def test_portfolio_risk():
    """Test portfolio risk management."""
    logging.info("Testing portfolio risk management...")
    try:
        df = generate_test_data()
        current_trades = [
            ('LONG', 100.0, 95.0, pd.Timestamp.now() - pd.Timedelta(days=1),
             pd.Timestamp.now())
        ]
        result = check_portfolio_risk(
            current_trades, 'SHORT', 1000.0, df
        )
        logging.info(f"Portfolio risk test passed. Result: {result}")
        return True
    except Exception as e:
        logging.error(f"Portfolio risk test failed: {e}")
        return False

def test_full_backtest():
    """Test the complete backtesting process."""
    logging.info("Testing full backtest...")
    try:
        # Generate sample data
        df = generate_test_data(200)
        
        # Prepare features
        df_processed = prepare_features(df, include_all=True)
        
        # Create and train a simple model for testing
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        X = df_processed[['close', 'volume']].fillna(0)
        y = np.where(df_processed['close'].pct_change().shift(-1) > 0, 2, 0)
        y = y[:-1]  # Remove last row as we don't have future data
        X = X[:-1]  # Match X and y sizes
        model.fit(X, y)
        
        # Run backtest
        balance, trades, balance_hist = realizar_backtest(df, model)
        
        logging.info(f"""
        Backtest completed successfully:
        Final Balance: {balance:.2f}
        Number of Trades: {len(trades)}
        Initial Balance: {balance_hist[0]:.2f}
        Final Balance: {balance_hist[-1]:.2f}
        """)
        return True
    except Exception as e:
        logging.error(f"Full backtest test failed: {e}")
        return False

def run_all_tests():
    """Run all test functions and report results."""
    tests = {
        "Signal Confirmation": test_signal_confirmation,
        "Position Sizing": test_position_sizing,
        "Stop Levels": test_stop_levels,
        "Portfolio Risk": test_portfolio_risk,
        "Full Backtest": test_full_backtest
    }
    
    results = {}
    for name, test_func in tests.items():
        logging.info(f"\nRunning test: {name}")
        try:
            result = test_func()
            results[name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            results[name] = f"❌ ERROR: {str(e)}"
            logging.error(f"Error in {name}: {e}", exc_info=True)
    
    # Print summary
    print("\n=== Test Results Summary ===")
    for name, result in results.items():
        print(f"{name}: {result}")

if __name__ == "__main__":
    run_all_tests()

