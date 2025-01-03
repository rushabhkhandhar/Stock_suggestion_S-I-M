import yfinance as yf
import pandas as pd
import numpy as np
import telegram
import asyncio
import pytz
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os
import sys
import json
import logging
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Optional
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot_log.txt')
    ]
)
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

@dataclass
class Trade:
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    risk: float
    reward: float
    strategy: str
    symbol: str

class RiskManagement:
    def __init__(self):
        self.min_rr_ratio = 2.0  # Minimum risk-reward ratio
        self.max_risk_per_trade = 0.01  # 1% max risk per trade
        self.required_edge = 0.5  # 50% minimum edge required
        
    def calculate_position_size(self, account_size: float, risk_per_trade: float, 
                              entry: float, stop_loss: float) -> float:
        """Calculate position size based on risk parameters"""
        risk_amount = account_size * risk_per_trade
        risk_per_share = abs(entry - stop_loss)
        return (risk_amount / risk_per_share) if risk_per_share > 0 else 0

    def optimize_risk_reward(self, entry: float, atr: float, 
                           support_levels: List[float], 
                           resistance_levels: List[float]) -> Tuple[float, float]:
        """
        Optimize stop loss and target based on ATR and key levels
        Returns: (stop_loss, target)
        """
        # Find nearest support for stop loss
        valid_supports = [s for s in support_levels if s < entry]
        stop_loss = max(valid_supports) if valid_supports else entry - (1.5 * atr)
        
        # Initial target based on minimum R:R
        min_target = entry + (abs(entry - stop_loss) * self.min_rr_ratio)
        
        # Find resistance above minimum target
        valid_resistances = [r for r in resistance_levels if r > min_target]
        target = min(valid_resistances) if valid_resistances else min_target
        
        # Ensure minimum R:R is maintained
        if (target - entry) / (entry - stop_loss) < self.min_rr_ratio:
            target = entry + (abs(entry - stop_loss) * self.min_rr_ratio)
            
        return stop_loss, target

class StrategyBacktester:
    def __init__(self):
        self.trades: List[Trade] = []
        self.risk_manager = RiskManagement()
        
    def backtest_strategy(self, df: pd.DataFrame, strategy_config: Dict) -> Dict:
        """
        Backtest a trading strategy and calculate performance metrics
        """
        signals = self._generate_signals(df, strategy_config)
        trades = self._execute_trades(df, signals)
        metrics = self._calculate_metrics(trades)
        
        return {
            'metrics': metrics,
            'trades': trades
        }
    
    def _generate_signals(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """Generate entry/exit signals based on strategy rules"""
        signals = pd.Series(index=df.index, data=0)
        
        # Entry conditions
        entry_conditions = [
            df['RSI'] > config.get('rsi_threshold', 50),
            df['Close'] > df['EMA_20'],
            df['Volume'] > df['Volume'].rolling(20).mean() * 1.2
        ]
        
        # Exit conditions
        exit_conditions = [
            df['RSI'] < config.get('rsi_exit', 30),
            df['Close'] < df['EMA_20']
        ]
        
        signals[np.all(entry_conditions, axis=0)] = 1
        signals[np.any(exit_conditions, axis=0)] = -1
        
        return signals
    
    def _execute_trades(self, df: pd.DataFrame, signals: pd.Series) -> List[Trade]:
        """Simulate trade execution with risk management"""
        trades = []
        in_position = False
        
        for i in range(1, len(df)):
            if not in_position and signals[i] == 1:
                entry_price = df['Close'].iloc[i]
                stop_loss, target = self.risk_manager.optimize_risk_reward(
                    entry_price,
                    df['ATR'].iloc[i],
                    df['Fib_support'].iloc[i:i+20].tolist(),
                    df['Fib_resistance'].iloc[i:i+20].tolist()
                )
                
                risk = entry_price - stop_loss
                reward = target - entry_price
                
                # Only take trades meeting minimum R:R
                if reward / risk >= self.risk_manager.min_rr_ratio:
                    entry_time = df.index[i]
                    in_position = True
                    
            elif in_position:
                exit_price = df['Close'].iloc[i]
                if (exit_price <= stop_loss or exit_price >= target or 
                    signals[i] == -1):
                    trades.append(Trade(
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_time=entry_time,
                        exit_time=df.index[i],
                        pnl=exit_price - entry_price,
                        risk=risk,
                        reward=reward,
                        strategy='momentum',
                        symbol=df.get('symbol', 'Unknown')
                    ))
                    in_position = False
        
        return trades
    
    def _calculate_metrics(self, trades: List[Trade]) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {}
            
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 0
        
        # Calculate system expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Calculate risk-adjusted metrics
        avg_risk = np.mean([t.risk for t in trades])
        edge = expectancy / avg_risk if avg_risk > 0 else 0
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'edge': edge,
            'profit_factor': avg_win / avg_loss if avg_loss > 0 else float('inf'),
            'avg_rr_ratio': np.mean([t.reward / t.risk for t in trades]),
            'sharpe_ratio': self._calculate_sharpe_ratio(trades),
            'max_drawdown': self._calculate_max_drawdown(trades)
        }
    
    def _calculate_sharpe_ratio(self, trades: List[Trade], risk_free_rate=0.05) -> float:
        """Calculate Sharpe ratio for the strategy"""
        daily_returns = pd.Series([t.pnl for t in trades])
        excess_returns = daily_returns.mean() - risk_free_rate/252
        return np.sqrt(252) * excess_returns / daily_returns.std() if len(trades) > 1 else 0
    
    def _calculate_max_drawdown(self, trades: List[Trade]) -> float:
        """Calculate maximum drawdown"""
        equity_curve = pd.Series([t.pnl for t in trades]).cumsum()
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve - rolling_max
        return abs(drawdowns.min()) if len(trades) > 0 else 0

class CrossValidator:
    """Perform walk-forward optimization and cross-validation"""
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        
    def walk_forward_optimization(self, df: pd.DataFrame, 
                                strategy_configs: List[Dict]) -> Dict:
        """
        Perform walk-forward optimization to find robust parameters
        """
        results = []
        split_size = len(df) // self.n_splits
        
        for i in range(self.n_splits - 1):
            train = df.iloc[i*split_size:(i+1)*split_size]
            test = df.iloc[(i+1)*split_size:(i+2)*split_size]
            
            # Test each configuration
            for config in strategy_configs:
                backtester = StrategyBacktester()
                train_results = backtester.backtest_strategy(train, config)
                test_results = backtester.backtest_strategy(test, config)
                
                # Check for robustness
                if (train_results['metrics']['edge'] > 0.5 and 
                    test_results['metrics']['edge'] > 0.4):
                    results.append({
                        'config': config,
                        'train_metrics': train_results['metrics'],
                        'test_metrics': test_results['metrics']
                    })
        
        # Return most robust configuration
        if results:
            return max(results, 
                      key=lambda x: x['test_metrics']['edge'])['config']
        return strategy_configs[0]  # Default to first config if none are robust

@dataclass
class Pattern:
    name: str
    strength: int
    description: str

class PatternRecognition:
    def __init__(self):
        self.window_size = 20

    def detect_candlestick_patterns(self, df) -> List[Dict]:
        patterns = []
        current_candle = {
            'open': df['Open'].iloc[-1],
            'high': df['High'].iloc[-1],
            'low': df['Low'].iloc[-1],
            'close': df['Close'].iloc[-1]
        }
        prev_candle = {
            'open': df['Open'].iloc[-2],
            'high': df['High'].iloc[-2],
            'low': df['Low'].iloc[-2],
            'close': df['Close'].iloc[-2]
        }

        # Detect engulfing pattern
        if self._is_bullish_engulfing(current_candle, prev_candle):
            patterns.append({
                'name': 'Bullish Engulfing',
                'strength': 2,
                'description': 'Strong bullish reversal'
            })
        elif self._is_bearish_engulfing(current_candle, prev_candle):
            patterns.append({
                'name': 'Bearish Engulfing',
                'strength': -2,
                'description': 'Strong bearish reversal'
            })

        # Detect doji
        if self._is_doji(current_candle):
            patterns.append({
                'name': 'Doji',
                'strength': 0,
                'description': 'Potential trend reversal'
            })

        # Detect hammer/shooting star
        if self._is_hammer(current_candle):
            patterns.append({
                'name': 'Hammer',
                'strength': 1,
                'description': 'Bullish reversal signal'
            })
        elif self._is_shooting_star(current_candle):
            patterns.append({
                'name': 'Shooting Star',
                'strength': -1,
                'description': 'Bearish reversal signal'
            })

        return patterns

    def detect_chart_patterns(self, df) -> List[Dict]:
        patterns = []
        
        # Double Top/Bottom detection
        if self._detect_double_top(df):
            patterns.append({
                'name': 'Double Top',
                'strength': -2,
                'description': 'Bearish reversal pattern'
            })
        elif self._detect_double_bottom(df):
            patterns.append({
                'name': 'Double Bottom',
                'strength': 2,
                'description': 'Bullish reversal pattern'
            })

        # Trend patterns
        trend_pattern = self._detect_trend_pattern(df)
        if trend_pattern:
            patterns.append(trend_pattern)

        return patterns

    def analyze_gaps(self, df) -> Optional[Dict]:
        try:
            current_open = df['Open'].iloc[-1]
            prev_close = df['Close'].iloc[-2]
            gap_size = ((current_open - prev_close) / prev_close) * 100
            
            if abs(gap_size) >= 1:  # 1% threshold
                return {
                    'type': 'Up Gap' if gap_size > 0 else 'Down Gap',
                    'size': abs(gap_size),
                    'strength': 1 if gap_size > 0 else -1,
                    'description': f"{'Bullish' if gap_size > 0 else 'Bearish'} gap of {abs(gap_size):.2f}%"
                }
        except Exception as e:
            logger.error(f"Error in gap analysis: {e}")
        return None

    def _is_doji(self, candle, threshold=0.1):
        body_size = abs(candle['close'] - candle['open'])
        total_size = candle['high'] - candle['low']
        return body_size <= (total_size * threshold)

    def _is_bullish_engulfing(self, current, previous):
        return (current['open'] < previous['close'] and 
                current['close'] > previous['open'] and
                current['close'] > current['open'])

    def _is_bearish_engulfing(self, current, previous):
        return (current['open'] > previous['close'] and 
                current['close'] < previous['open'] and
                current['close'] < current['open'])

    def _is_hammer(self, candle):
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        return (lower_shadow > (2 * body_size) and 
                upper_shadow < body_size)

    def _is_shooting_star(self, candle):
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        return (upper_shadow > (2 * body_size) and 
                lower_shadow < body_size)

    def _detect_double_top(self, df, threshold=0.02):
        highs = df['High'].rolling(window=5).max()
        peaks = self._find_peaks(highs.values)
        if len(peaks) >= 2:
            peak1, peak2 = peaks[-2], peaks[-1]
            return (abs(highs[peak1] - highs[peak2]) / highs[peak1]) < threshold

    def _detect_double_bottom(self, df, threshold=0.02):
        lows = df['Low'].rolling(window=5).min()
        troughs = self._find_troughs(lows.values)
        if len(troughs) >= 2:
            trough1, trough2 = troughs[-2], troughs[-1]
            return (abs(lows[trough1] - lows[trough2]) / lows[trough1]) < threshold

    def _detect_trend_pattern(self, df):
        closes = df['Close'].values[-20:]
        x = np.arange(len(closes))
        slope, _, r_value, _, _ = stats.linregress(x, closes)
        
        if abs(r_value) > 0.7:  # Strong trend
            if slope > 0:
                return {
                    'name': 'Strong Uptrend',
                    'strength': 2,
                    'description': 'Consistent price increase'
                }
            else:
                return {
                    'name': 'Strong Downtrend',
                    'strength': -2,
                    'description': 'Consistent price decrease'
                }
        return None

    @staticmethod
    def _find_peaks(x, distance=5):
        peaks = []
        for i in range(distance, len(x) - distance):
            if all(x[i] > x[i - j] for j in range(1, distance + 1)) and \
               all(x[i] > x[i + j] for j in range(1, distance + 1)):
                peaks.append(i)
        return peaks

    @staticmethod
    def _find_troughs(x, distance=5):
        troughs = []
        for i in range(distance, len(x) - distance):
            if all(x[i] < x[i - j] for j in range(1, distance + 1)) and \
               all(x[i] < x[i + j] for j in range(1, distance + 1)):
                troughs.append(i)
        return troughs


class StockScreener:
    def __init__(self, telegram_token, telegram_chat_id, index_symbol='^NSEI'):
        """Initialize the stock screener with Telegram credentials"""
        self.index_symbol = index_symbol
        self.stocks = None
        self.results = {}
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.bot = telegram.Bot(token=self.telegram_token)
        self.screened_signals = set()
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.last_signals = self.load_last_signals()
        self.pattern_recognition = PatternRecognition()
        self.risk_manager = RiskManagement()
        self.backtester = StrategyBacktester()
        self.validator = CrossValidator()
        
    def get_nifty_stocks(self):
        """Get list of stocks to monitor"""
        self.stocks = [
                    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
                "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
                "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "HCLTECH.NS", "BAJFINANCE.NS", 
                "WIPRO.NS", "MARUTI.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "TITAN.NS", 
                "TECHM.NS", "SUNPHARMA.NS", "M&M.NS", "ADANIGREEN.NS", "POWERGRID.NS", 
                "NTPC.NS", "ONGC.NS", "BPCL.NS", "INDUSINDBK.NS", "GRASIM.NS", 
                "ADANIPORTS.NS", "JSWSTEEL.NS", "COALINDIA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", 
                "EICHERMOT.NS", "BAJAJFINSV.NS", "TATAMOTORS.NS", "DIVISLAB.NS", "HDFCLIFE.NS",
                "CIPLA.NS", "HEROMOTOCO.NS", "SBICARD.NS", "ADANIENT.NS", "UPL.NS", 
                "BRITANNIA.NS", "ICICIPRULI.NS", "SHREECEM.NS", "PIDILITIND.NS", "DMART.NS",
                "ABB.NS", "AIAENG.NS", "ALKEM.NS",  "AMBUJACEM.NS", 
                "AUROPHARMA.NS", "BANDHANBNK.NS", "BERGEPAINT.NS", "BOSCHLTD.NS", "CANBK.NS", 
                "CHOLAFIN.NS", "CUMMINSIND.NS", "DABUR.NS", "DLF.NS", "ESCORTS.NS", 
                "FEDERALBNK.NS", "GLAND.NS", "GLAXO.NS", "GODREJCP.NS", "GODREJPROP.NS", 
                "HAL.NS", "HAVELLS.NS", "IGL.NS", "IRCTC.NS", "LICI.NS", 
                "LUPIN.NS",  "MRF.NS", "NAUKRI.NS", 
                "PEL.NS", "PFC.NS", "PNB.NS", "RECLTD.NS", "SIEMENS.NS", 
                "SRF.NS", "TATACHEM.NS", "TATAELXSI.NS", "TRENT.NS", "TVSMOTOR.NS", 
                "VBL.NS", "VEDL.NS", "WHIRLPOOL.NS", "ZOMATO.NS","INOXWIND.NS","SOLARA.NS","INOXGREEN.NS","MOTHERSON.NS",
                "LLOYDSENGG.NS","HCC.NS","CAMLINFINE.NS","AURUM.NS","AXISCADES.NS"
                ]
                
    def get_stock_data(self, symbol, duration='1mo', interval='5m'):
        """
        Fetch historical data for a stock
        duration: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
        interval: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        """
        try:
            stock = yf.Ticker(symbol)
            # For intraday data, we can only get 5d max
            if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
                duration = '5d'
            
            df = stock.history(period=duration, interval=interval)
            
            if df.empty:
                logger.info(f"No data available for {symbol}")
                return None
                
            return df
            
        except Exception as e:
            logger.info(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_indicators(self, df):
        """Custom indicator calculations including pivot points and fibonacci levels"""
        try:
            # Previous EMAs
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            # Calculate Pivot Points
            df['PP'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
            df['R1'] = 2 * df['PP'] - df['Low'].shift(1)
            df['S1'] = 2 * df['PP'] - df['High'].shift(1)
            df['R2'] = df['PP'] + (df['High'].shift(1) - df['Low'].shift(1))
            df['S2'] = df['PP'] - (df['High'].shift(1) - df['Low'].shift(1))
            df['R3'] = df['High'].shift(1) + 2 * (df['PP'] - df['Low'].shift(1))
            df['S3'] = df['Low'].shift(1) - 2 * (df['High'].shift(1) - df['PP'])
            
            # Calculate Fibonacci Levels (using last swing high and low)
            lookback = 20  # Period to find swing high/low
            high = df['High'].rolling(window=lookback, center=False).max()
            low = df['Low'].rolling(window=lookback, center=False).min()
            
            # Fibonacci retracement levels
            diff = high - low
            df['Fib_0'] = high
            df['Fib_236'] = high - (diff * 0.236)
            df['Fib_382'] = high - (diff * 0.382)
            df['Fib_500'] = high - (diff * 0.500)
            df['Fib_618'] = high - (diff * 0.618)
            df['Fib_786'] = high - (diff * 0.786)
            df['Fib_100'] = low
            
            # Previous RSI and other calculations remain the same
            delta = df['Close'].diff()
            gains = delta.clip(lower=0)
            losses = -delta.clip(upper=0)
            avg_gain = gains.rolling(window=14).mean()
            avg_loss = losses.rolling(window=14).mean()
            relative_strength = avg_gain / avg_loss
            df['RSI'] = 100.0 - (100.0 / (1.0 + relative_strength))
            
            # Previous indicators remain the same
            df['ATR'] = np.maximum(
                df['High'] - df['Low'],
                np.abs(df['High'] - df['Close'].shift(1)),
                np.abs(df['Low'] - df['Close'].shift(1))
            ).rolling(window=14).mean()
            
            df['Trend_Strength'] = ((df['Close'] - df['EMA_20']) / df['EMA_20']) * 100
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            return df
        
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return None

    def analyze_stock(self, symbol):
        """
        Analyze a single stock for trading opportunities with enhanced risk management
        and strategy validation
        """
        try:
            # Get data for different timeframes
            df_intraday = self.get_stock_data(symbol, duration='5d', interval='5m')
            df_daily = self.get_stock_data(symbol, duration='1mo', interval='1d')
            df_backtest = self.get_stock_data(symbol, duration='1y', interval='1d')  # For backtesting
            
            if df_intraday is None or df_daily is None or df_backtest is None:
                return None
                
            # Calculate indicators for all timeframes
            df_intraday = self.calculate_indicators(df_intraday)
            df_daily = self.calculate_indicators(df_daily)
            df_backtest = self.calculate_indicators(df_backtest)
            
            if df_intraday is None or df_daily is None or df_backtest is None:
                return None
                
            current_close = df_intraday['Close'].iloc[-1]
            current_atr = df_daily['ATR'].iloc[-1]
            
            # Get pattern recognition signals
            patterns = {
                'candlestick': self.pattern_recognition.detect_candlestick_patterns(df_daily),
                'chart': self.pattern_recognition.detect_chart_patterns(df_daily),
                'gap': self.pattern_recognition.analyze_gaps(df_daily)
            }
            
            # Calculate pattern score
            pattern_score = sum(p['strength'] for p in patterns['candlestick']) + \
                        sum(p['strength'] for p in patterns['chart'])
            if patterns['gap']:
                pattern_score += patterns['gap']['strength']
            
            # Backtest strategies to validate edge
            strategy_metrics = {}
            validated_signals = {}
            
            for strategy in ['intraday', 'swing', 'momentum']:
                # Get initial signals
                signal = getattr(self, f'get_{strategy}_signals')(
                    df_intraday if strategy == 'intraday' else df_daily
                )
                
                if signal and signal['signal'] == 'buy':
                    # Backtest the strategy
                    backtest_config = {
                        'strategy': strategy,
                        'rsi_threshold': 50,
                        'volume_multiplier': 1.2,
                        'pattern_score': pattern_score
                    }
                    
                    results = self.backtester.backtest_strategy(df_backtest, backtest_config)
                    metrics = results['metrics']
                    
                    # Validate strategy edge
                    if metrics.get('edge', 0) > self.risk_manager.required_edge:
                        # Optimize risk-reward based on ATR and key levels
                        stop_loss, target = self.risk_manager.optimize_risk_reward(
                            current_close,
                            current_atr,
                            [level for level in df_daily['Fib_support'].iloc[-20:] if level < current_close],
                            [level for level in df_daily['Fib_resistance'].iloc[-20:] if level > current_close]
                        )
                        
                        # Calculate risk-reward ratio
                        risk = current_close - stop_loss
                        reward = target - current_close
                        rr_ratio = reward / risk if risk > 0 else 0
                        
                        # Only include signals with acceptable R:R
                        if rr_ratio >= self.risk_manager.min_rr_ratio:
                            signal.update({
                                'suggested_entry': current_close,
                                'suggested_sl': stop_loss,
                                'suggested_target': target,
                                'risk_reward_ratio': rr_ratio,
                                'expected_value': metrics['expectancy'],
                                'win_rate': metrics['win_rate'],
                                'avg_win': metrics['avg_win'],
                                'avg_loss': metrics['avg_loss']
                            })
                            
                            # Adjust signal strength based on pattern score
                            signal['strength'] += min(max(pattern_score // 2, -2), 2)
                            if pattern_score > 0:
                                signal['reasons'].append("Bullish pattern confirmed")
                            elif pattern_score < 0:
                                signal['reasons'].append("Bearish pattern detected")
                            
                            validated_signals[strategy] = signal
                            strategy_metrics[strategy] = metrics
            
            # Return None if no valid signals found
            if not validated_signals:
                return None
                
            return {
                'symbol': symbol,
                'close': current_close,
                'volume': df_intraday['Volume'].iloc[-1],
                'signals': validated_signals,
                'patterns': patterns,
                'metrics': strategy_metrics,
                'atr': current_atr
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def get_fib_support_resistance(self, df, current_price):
        """Get nearest Fibonacci support and resistance levels"""
        fib_levels = {
            'support': [],
            'resistance': []
        }
        
        for level in ['Fib_236', 'Fib_382', 'Fib_500', 'Fib_618', 'Fib_786']:
            price = df[level].iloc[-1]
            if price < current_price:
                fib_levels['support'].append(price)
            else:
                fib_levels['resistance'].append(price)
                
        return {
            'nearest_support': max(fib_levels['support']) if fib_levels['support'] else None,
            'nearest_resistance': min(fib_levels['resistance']) if fib_levels['resistance'] else None
        }


    def get_intraday_signals(self, df):
        """
        Enhanced Intraday Strategy with Pivot Points and Fibonacci:
        - Breakout above 20 EMA
        - RSI > 50
        - Volume confirmation
        - Pivot point and Fibonacci support/resistance
        """
        try:
            current_close = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2]
            ema_20 = df['EMA_20'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            
            # Get pivot points
            pp = df['PP'].iloc[-1]
            r1 = df['R1'].iloc[-1]
            s1 = df['S1'].iloc[-1]
            r2 = df['R2'].iloc[-1]
            s2 = df['S2'].iloc[-1]
            
            # Get Fibonacci levels
            fib_levels = self.get_fib_support_resistance(df, current_close)
            
            signal_strength = 0
            reasons = []
            
            # Previous conditions
            if current_close > ema_20 and prev_close <= df['EMA_20'].iloc[-2]:
                signal_strength += 1
                reasons.append("Breakout above 20 EMA")
            
            if rsi > 50:
                signal_strength += 1
                reasons.append("RSI > 50")
            
            if volume_ratio > 1.2:
                signal_strength += 1
                reasons.append("High volume confirmation")
            
            # Pivot point conditions
            if current_close > pp and prev_close <= pp:
                signal_strength += 1
                reasons.append("Breakout above Pivot Point")
            
            # Fibonacci conditions
            if fib_levels['nearest_support']:
                if abs(current_close - fib_levels['nearest_support']) / current_close < 0.01:
                    signal_strength += 1
                    reasons.append("Price near Fibonacci support")
                    
            # Calculate optimal stop loss and target using both pivot and fib levels
            atr = df['ATR'].iloc[-1]
            suggested_sl = current_close - atr
            suggested_target = current_close + (atr * 2)
            
            # Adjust based on nearest support/resistance
            if fib_levels['nearest_support']:
                suggested_sl = max(suggested_sl, fib_levels['nearest_support'])
            if fib_levels['nearest_resistance']:
                suggested_target = min(suggested_target, fib_levels['nearest_resistance'])
            
            return {
                'signal': 'buy' if signal_strength >= 3 else 'neutral',
                'strength': signal_strength,
                'reasons': reasons,
                'suggested_entry': current_close,
                'suggested_sl': suggested_sl,
                'suggested_target': suggested_target,
                'pivot_points': {
                    'PP': pp,
                    'R1': r1,
                    'S1': s1,
                    'R2': r2,
                    'S2': s2
                },
                'fibonacci_levels': {
                    'support': fib_levels['nearest_support'],
                    'resistance': fib_levels['nearest_resistance']
                }
            }
            
        except Exception as e:
            logger.info(f"Error in intraday signals: {str(e)}")
            return None

    def get_swing_signals(self, df):
        """
        Enhanced Swing Strategy with Fibonacci:
        - Pullback to 10 EMA
        - Low RSI (30-45)
        - Overall uptrend
        - Fibonacci retracement support
        """
        try:
            current_close = df['Close'].iloc[-1]
            ema_10 = df['EMA_10'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            
            # Get Fibonacci levels
            fib_levels = self.get_fib_support_resistance(df, current_close)
            
            signal_strength = 0
            reasons = []
            
            # Previous conditions
            price_to_ema = abs((current_close - ema_10) / ema_10) * 100
            if price_to_ema <= 0.5:
                signal_strength += 1
                reasons.append("Price near 10 EMA")
            
            if 30 <= rsi <= 45:
                signal_strength += 1
                reasons.append("RSI indicates pullback")
                
            if df['Trend_Strength'].iloc[-1] > 0:
                signal_strength += 1
                reasons.append("Overall uptrend intact")
            
            # Fibonacci conditions
            if fib_levels['nearest_support']:
                if abs(current_close - fib_levels['nearest_support']) / current_close < 0.02:
                    signal_strength += 2
                    reasons.append("Price at key Fibonacci support")
            
            atr = df['ATR'].iloc[-1]
            suggested_sl = current_close - (atr * 1.5)
            suggested_target = current_close + (atr * 3)
            
            # Adjust based on Fibonacci levels
            if fib_levels['nearest_support']:
                suggested_sl = max(suggested_sl, fib_levels['nearest_support'])
            if fib_levels['nearest_resistance']:
                suggested_target = min(suggested_target, fib_levels['nearest_resistance'])
            
            return {
                'signal': 'buy' if signal_strength >= 3 else 'neutral',
                'strength': signal_strength,
                'reasons': reasons,
                'suggested_entry': current_close,
                'suggested_sl': suggested_sl,
                'suggested_target': suggested_target,
                'fibonacci_levels': {
                    'support': fib_levels['nearest_support'],
                    'resistance': fib_levels['nearest_resistance']
                }
            }
            
        except Exception as e:
            logger.info(f"Error in swing signals: {str(e)}")
            return None

    def get_momentum_signals(self, df):
        """
        Enhanced Momentum Strategy with Fibonacci:
        - 10 EMA crossover
        - RSI > 60
        - Strong volume and trend
        - Fibonacci breakout confirmation
        """
        try:
            current_close = df['Close'].iloc[-1]
            ema_10 = df['EMA_10'].iloc[-1]
            ema_20 = df['EMA_20'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            
            # Get Fibonacci levels
            fib_levels = self.get_fib_support_resistance(df, current_close)
            
            signal_strength = 0
            reasons = []
            
            # Previous conditions
            if (ema_10 > ema_20 and 
                df['EMA_10'].iloc[-2] <= df['EMA_20'].iloc[-2]):
                signal_strength += 2
                reasons.append("10 EMA crossed above 20 EMA")
            
            if rsi > 60:
                signal_strength += 1
                reasons.append("RSI > 60")
                
            if volume_ratio > 1.5:
                signal_strength += 1
                reasons.append("Strong volume confirmation")
                
            if df['Trend_Strength'].iloc[-1] > 2:
                signal_strength += 1
                reasons.append("Strong uptrend")
            
            # Fibonacci breakout condition
            if fib_levels['nearest_resistance']:
                if current_close > fib_levels['nearest_resistance']:
                    signal_strength += 2
                    reasons.append("Breakout above Fibonacci resistance")
            
            atr = df['ATR'].iloc[-1]
            suggested_sl = current_close - (atr * 2)
            suggested_target = current_close + (atr * 4)
            
            # Adjust based on Fibonacci levels
            if fib_levels['nearest_support']:
                suggested_sl = max(suggested_sl, fib_levels['nearest_support'])
            
            return {
                'signal': 'buy' if signal_strength >= 4 else 'neutral',  # Higher threshold for momentum
                'strength': signal_strength,
                'reasons': reasons,
                'suggested_entry': current_close,
                'suggested_sl': suggested_sl,
                'suggested_target': suggested_target,
                'fibonacci_levels': {
                    'support': fib_levels['nearest_support'],
                    'resistance': fib_levels['nearest_resistance']
                }
            }
            
        except Exception as e:
            logger.info(f"Error in momentum signals: {str(e)}")
            return None

    def is_market_open(self):
        """Check if Indian market is open"""
        now = datetime.now(self.ist_tz)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            return False
            
        # Market hours: 9:15 AM to 3:30 PM IST
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end

    async def send_telegram_alert(self, message):
        """Send alert to Telegram with minimal rate limiting"""
        try:
            await asyncio.sleep(0.2)  # 200ms delay between messages
            await self.bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            
    def format_signal_message(self, stock, strategy):
        """Enhanced format signal message with pattern information"""
        message = f"🔔 {strategy.upper()} Trading Signal\n\n"
        message += f"Stock: {stock['symbol']}\n"
        message += f"Current Price: ₹{stock['close']:.2f}\n"
        message += f"Signal Strength: {stock['signal']['strength']}\n"
        message += f"Entry: ₹{stock['signal']['suggested_entry']:.2f}\n"
        message += f"Stop Loss: ₹{stock['signal']['suggested_sl']:.2f}\n"
        message += f"Target: ₹{stock['signal']['suggested_target']:.2f}\n\n"

        if 'metrics' in stock and strategy in stock['metrics']:
            metrics = stock['metrics'][strategy]
            message += "📈 Backtest Performance:\n"
            message += f"Win Rate: {metrics['win_rate']:.1%}\n"
            message += f"Total Trades: {metrics['total_trades']}\n"
            message += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
            message += f"Edge: {metrics.get('edge', 0):.2f}\n"
            message += f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%\n\n"
            
        if 'pivot_points' in stock['signal']:
            message += "Pivot Points:\n"
            message += f"PP: ₹{stock['signal']['pivot_points']['PP']:.2f}\n"
            message += f"R1: ₹{stock['signal']['pivot_points']['R1']:.2f}\n"
            message += f"S1: ₹{stock['signal']['pivot_points']['S1']:.2f}\n"
            message += f"R2: ₹{stock['signal']['pivot_points']['R2']:.2f}\n"
            message += f"S2: ₹{stock['signal']['pivot_points']['S2']:.2f}\n\n"
        
        if 'fibonacci_levels' in stock['signal']:
            message += "Fibonacci Levels:\n"
            if stock['signal']['fibonacci_levels']['support']:
                message += f"Support: ₹{stock['signal']['fibonacci_levels']['support']:.2f}\n"
            if stock['signal']['fibonacci_levels']['resistance']:
                message += f"Resistance: ₹{stock['signal']['fibonacci_levels']['resistance']:.2f}\n\n"
        
        if 'patterns' in stock and (stock['patterns']['candlestick'] or 
                                stock['patterns']['chart'] or 
                                stock['patterns']['gap']):
            message += "📊 Pattern Analysis:\n\n"
            
            if stock['patterns']['candlestick']:
                message += "Candlestick Patterns:\n"
                for pattern in stock['patterns']['candlestick'][:2]:
                    message += f"- {pattern['name']}: {pattern['description']}\n"
                message += "\n"
            
            if stock['patterns']['chart']:
                message += "Chart Patterns:\n"
                for pattern in stock['patterns']['chart']:
                    message += f"- {pattern['name']}: {pattern['description']}\n"
                message += "\n"
            
            if stock['patterns']['gap']:
                gap = stock['patterns']['gap']
                message += f"Gap Analysis: {gap['type']} ({gap['size']:.2f}%)\n\n"
        
        message += "Reasons:\n"
        message += "\n".join(f"- {reason}" for reason in stock['signal']['reasons'])
        
        return message

    def scan_market(self):
        """Scan all stocks and generate trading opportunities"""
        self.get_nifty_stocks()
          
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(self.analyze_stock, self.stocks))
            
        # Filter out None results and organize by strategy
        valid_results = [r for r in results if r is not None]
        
        opportunities = {
            'intraday': [],
            'swing': [],
            'momentum': []
        }
        
        for result in valid_results:
            for strategy in ['intraday', 'swing', 'momentum']:
                if result['signals'][strategy] is not None and result['signals'][strategy]['signal'] == 'buy':
                    opportunities[strategy].append({
                        'symbol': result['symbol'],
                        'close': result['close'],
                        'signal': result['signals'][strategy],
                        'volume': result['volume'],
                        'patterns': result['patterns'],
                        
                    })
                    
        return opportunities
    
    def load_last_signals(self):
        """Load last signals from file"""
        try:
            if os.path.exists('last_signals.json'):
                with open('last_signals.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading last signals: {e}")
        return {}
        
    def save_last_signals(self, signals):
        """Save signals to file"""
        try:
            with open('last_signals.json', 'w') as f:
                json.dump(signals, f)
        except Exception as e:
            logger.error(f"Error saving signals: {e}")

    def has_significant_changes(self, stock, strategy, new_signal):
        """Check if there are significant changes in the signal"""
        key = f"{stock}_{strategy}"
        last_signal = self.last_signals.get(key, {})
        
        # If this stock wasn't in the previous scan, it's definitely a significant change
        if not last_signal or 'alert_time' not in last_signal:
            logger.info(f"New stock detected: {stock} with strategy {strategy}")
            new_signal['alert_time'] = datetime.now(self.ist_tz).isoformat()
            return True
            
        # Rest of your existing significant changes checks
        significant_changes = [
            abs((new_signal['suggested_entry'] - last_signal.get('suggested_entry', 0)) / 
                last_signal.get('suggested_entry', 1) * 100) > 0.2,
            new_signal['strength'] != last_signal.get('strength', 0),
            set(new_signal['reasons']) != set(last_signal.get('reasons', [])),
            new_signal['signal'] != last_signal.get('signal', 'neutral')
        ]
        
        # Time-based check
        last_alert_time = last_signal.get('alert_time')
        if last_alert_time:
            last_alert_time = datetime.fromisoformat(last_alert_time)
            time_diff = datetime.now(self.ist_tz) - last_alert_time
            if time_diff.total_seconds() < 300:  # 5 minutes minimum between alerts
                return False
        
        new_signal['alert_time'] = datetime.now(self.ist_tz).isoformat()
        return any(significant_changes)

    async def scan_and_alert(self):
        if not self.is_market_open():
            logger.info("Market is closed")
            return

        opportunities = self.scan_market()
        new_signals = {}
        messages = {}
        
        # Initialize messages dict for each strategy
        for strategy in opportunities.keys():
            messages[strategy] = []

        # Start message
        start_message = f"🔄 Market Scan Started at {datetime.now(self.ist_tz).strftime('%H:%M:%S')}\n"
        await self.send_telegram_alert(start_message)
        
        # Process each strategy's signals
        for strategy, stocks in opportunities.items():
            if stocks:
                messages[strategy].append(f"\n🔍 {strategy.upper()} SIGNALS\n")
                
                sorted_stocks = sorted(stocks, key=lambda x: x['signal']['strength'], reverse=True)
                
                # Batch stock signals for this strategy
                for stock in sorted_stocks:
                    if stock['signal']['signal'] == 'buy':
                        key = f"{stock['symbol']}_{strategy}"
                        if self.has_significant_changes(stock['symbol'], strategy, stock['signal']):
                            messages[strategy].append(self.format_signal_message(stock, strategy))
                            new_signals[key] = stock['signal']

        # Send batched messages for each strategy
        for strategy, strategy_messages in messages.items():
            if strategy_messages:
                combined_message = "\n\n".join(strategy_messages)
                if len(combined_message.strip()) > 0:
                    await self.send_telegram_alert(combined_message)
                    await asyncio.sleep(0.2)  # Small delay between strategies

        # Send summary
        summary = (f"\n✨ Scan completed at {datetime.now(self.ist_tz).strftime('%H:%M:%S')}\n"
                    f"Total signals found: {len(new_signals)}")
        await self.send_telegram_alert(summary)
        
        # Update and save signals
        self.last_signals = new_signals  
        self.save_last_signals(new_signals)

async def main():
    """Main async function optimized for GitHub Actions"""
    try:
        # Get tokens from environment variables
        TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
        TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
        
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            logger.error("Telegram credentials not found!")
            return
        
        # Initialize screener
        screener = StockScreener(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        
        # Check if market is open
        if not screener.is_market_open():
            logger.info("Market is closed. Exiting.")
            return
            
        try:
            # Perform single market scan and send alerts
            await screener.scan_and_alert()
            logger.info("Market scan completed successfully")
        
        except Exception as e:
            error_message = f"❌ Error during market scan: {str(e)}"
            logger.error(error_message)
            await screener.send_telegram_alert(error_message)
    
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")

def run_screener():
    """Function to run the async event loop"""
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Unhandled error in run_screener: {e}")

if __name__ == "__main__":
    run_screener()