# utils/indicators.py

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import StochasticOscillator, RSIIndicator, ROCIndicator
from ta.trend import ADXIndicator, MACD, EMAIndicator, IchimokuIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator, AccDistIndexIndicator
from ta.others import DailyReturnIndicator

class AdvancedIndicators:
    """Advanced technical indicators for cryptocurrency trading analysis."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame containing OHLCV data.
        
        Args:
            df: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.df = df.copy()
        self._validate_dataframe()
        self._initialize_price_metrics()
    
    def _validate_dataframe(self) -> None:
        """Validate that the DataFrame has required columns."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    def _initialize_price_metrics(self) -> None:
        """Initialize basic price metrics used by various indicators."""
        # Basic returns
        self.df['returns'] = self.df['close'].pct_change()
        self.df['log_returns'] = np.log1p(self.df['returns'])
        
        # Volatility measurement
        self.df['true_range'] = pd.DataFrame({
            'hl': self.df['high'] - self.df['low'],
            'hc': abs(self.df['high'] - self.df['close'].shift(1)),
            'lc': abs(self.df['low'] - self.df['close'].shift(1))
        }).max(axis=1)

    def add_volatility_indicators(self, 
                                bb_window: int = 20, 
                                bb_dev: float = 2.0,
                                atr_window: int = 14) -> pd.DataFrame:
        """
        Add volatility-based indicators.
        
        Args:
            bb_window: Bollinger Bands period
            bb_dev: Bollinger Bands standard deviation
            atr_window: ATR period
        """
        # Bollinger Bands
        bb = BollingerBands(
            close=self.df['close'],
            window=bb_window,
            window_dev=bb_dev
        )
        self.df['bb_high'] = bb.bollinger_hband()
        self.df['bb_mid'] = bb.bollinger_mavg()
        self.df['bb_low'] = bb.bollinger_lband()
        self.df['bb_width'] = (self.df['bb_high'] - self.df['bb_low']) / self.df['bb_mid']
        self.df['bb_percent'] = (self.df['close'] - self.df['bb_low']) / (self.df['bb_high'] - self.df['bb_low'])

        # ATR and Normalized ATR
        atr = AverageTrueRange(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            window=atr_window
        )
        self.df['atr'] = atr.average_true_range()
        self.df['natr'] = self.df['atr'] / self.df['close'] * 100  # Normalized ATR

        return self.df

    def add_momentum_indicators(self,
                              rsi_window: int = 14,
                              stoch_window: int = 14,
                              stoch_smooth: int = 3,
                              roc_window: int = 12) -> pd.DataFrame:
        """
        Add momentum-based indicators.
        
        Args:
            rsi_window: RSI calculation period
            stoch_window: Stochastic Oscillator period
            stoch_smooth: Stochastic smoothing period
            roc_window: Rate of Change period
        """
        # RSI
        rsi = RSIIndicator(close=self.df['close'], window=rsi_window)
        self.df['rsi'] = rsi.rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            window=stoch_window,
            smooth_window=stoch_smooth
        )
        self.df['stoch_k'] = stoch.stoch()
        self.df['stoch_d'] = stoch.stoch_signal()
        
        # Rate of Change
        roc = ROCIndicator(close=self.df['close'], window=roc_window)
        self.df['roc'] = roc.roc()
        
        # Custom momentum indicators
        self.df['momentum'] = self.df['close'].diff(roc_window)
        self.df['mom_ma'] = self.df['momentum'].rolling(window=roc_window).mean()

        return self.df

    def add_trend_indicators(self,
                           macd_fast: int = 12,
                           macd_slow: int = 26,
                           macd_signal: int = 9,
                           adx_window: int = 14) -> pd.DataFrame:
        """
        Add trend-following indicators.
        
        Args:
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            adx_window: ADX period
        """
        # MACD
        macd = MACD(
            close=self.df['close'],
            window_slow=macd_slow,
            window_fast=macd_fast,
            window_sign=macd_signal
        )
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_diff'] = macd.macd_diff()
        
        # ADX
        adx = ADXIndicator(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            window=adx_window
        )
        self.df['adx'] = adx.adx()
        self.df['di_plus'] = adx.adx_pos()
        self.df['di_minus'] = adx.adx_neg()

        # Ichimoku Cloud
        ichimoku = IchimokuIndicator(
            high=self.df['high'],
            low=self.df['low']
        )
        self.df['ichimoku_a'] = ichimoku.ichimoku_a()
        self.df['ichimoku_b'] = ichimoku.ichimoku_b()
        
        return self.df

    def add_volume_indicators(self) -> pd.DataFrame:
        """Add volume-based indicators."""
        # On-Balance Volume
        obv = OnBalanceVolumeIndicator(
            close=self.df['close'],
            volume=self.df['volume']
        )
        self.df['obv'] = obv.on_balance_volume()
        
        # Volume-Price Trend
        vpt = VolumePriceTrendIndicator(
            close=self.df['close'],
            volume=self.df['volume']
        )
        self.df['vpt'] = vpt.volume_price_trend()
        
        # Accumulation/Distribution Index
        adi = AccDistIndexIndicator(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            volume=self.df['volume']
        )
        self.df['adi'] = adi.acc_dist_index()
        
        # Custom volume indicators
        self.df['volume_sma'] = self.df['volume'].rolling(window=20).mean()
        self.df['volume_std'] = self.df['volume'].rolling(window=20).std()
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma']
        
        return self.df

    def add_custom_features(self) -> pd.DataFrame:
        """Add custom calculated features specific to crypto trading."""
        # Price patterns
        self.df['price_std'] = self.df['close'].rolling(window=20).std()
        self.df['price_z_score'] = (self.df['close'] - self.df['close'].rolling(window=20).mean()) / self.df['price_std']
        
        # Volatility patterns
        self.df['volatility_ratio'] = self.df['true_range'] / self.df['true_range'].rolling(window=20).mean()
        
        # Volume patterns
        self.df['vol_price_impact'] = self.df['returns'].abs() / (self.df['volume'] * self.df['close'])
        
        # Time-based features (assuming index is datetime)
        if isinstance(self.df.index, pd.DatetimeIndex):
            self.df['hour'] = self.df.index.hour
            self.df['day_of_week'] = self.df.index.dayofweek
            
        return self.df

    def add_all_indicators(self, include_custom: bool = True) -> pd.DataFrame:
        """
        Add all available indicators with default parameters.
        
        Args:
            include_custom: Whether to include custom features
        """
        self.add_volatility_indicators()
        self.add_momentum_indicators()
        self.add_trend_indicators()
        self.add_volume_indicators()
        
        if include_custom:
            self.add_custom_features()
        
        # Clean up NaN values
        self.df = self.df.fillna(method='bfill').fillna(method='ffill')
        
        return self.df

def prepare_features(df: pd.DataFrame, 
                    include_all: bool = True,
                    custom_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Prepare a DataFrame with technical indicators for the trading model.
    
    Args:
        df: Input DataFrame with OHLCV data
        include_all: Whether to include all available indicators
        custom_config: Optional dictionary with custom parameters for indicators
        
    Returns:
        DataFrame with added technical indicators
    """
    indicators = AdvancedIndicators(df)
    
    if include_all:
        return indicators.add_all_indicators()
    
    # If not including all, use custom configuration
    if custom_config:
        if 'volatility' in custom_config:
            indicators.add_volatility_indicators(**custom_config['volatility'])
        if 'momentum' in custom_config:
            indicators.add_momentum_indicators(**custom_config['momentum'])
        if 'trend' in custom_config:
            indicators.add_trend_indicators(**custom_config['trend'])
        if 'volume' in custom_config:
            indicators.add_volume_indicators()
    
    return indicators.df.fillna(method='bfill').fillna(method='ffill')

