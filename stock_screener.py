import yfinance as yf
import pandas as pd
import numpy as np

import telegram
import asyncio
import pytz
from datetime import datetime,timedelta
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os
import sys
import json
import logging
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
        self.intraday_cache = {}
        
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
                "VBL.NS", "VEDL.NS", "WHIRLPOOL.NS", "ZOMATO.NS","INOXWIND.NS"
                ]
                
    def get_stock_data(self, symbol, duration='1mo', interval='1m'):
        """
        Enhanced stock data fetching with better intraday handling
        """
        try:
            stock = yf.Ticker(symbol)
            
            # For intraday, fetch most recent data possible
            if interval in ['1m', '2m', '5m']:
                duration = '1d'  # Get today's data for more frequent updates
                
            df = stock.history(period=duration, interval=interval)
            
            if df.empty:
                logger.info(f"No data available for {symbol}")
                return None
                
            # Add timestamp for intraday caching
            df['timestamp'] = df.index
            return df
            
        except Exception as e:
            logger.info(f"Error fetching data for {symbol}: {str(e)}")
            return None


    def calculate_indicators(self, df):
        """Calculate technical indicators with optimized parameters for intraday"""
        try:
            # Faster EMAs for intraday
            df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            # Optimized RSI calculation
            delta = df['Close'].diff()
            gains = delta.clip(lower=0)
            losses = -delta.clip(upper=0)
            
            # Shorter RSI period for intraday
            rsi_period = 9
            avg_gain = gains.rolling(window=rsi_period).mean()
            avg_loss = losses.rolling(window=rsi_period).mean()
            
            relative_strength = avg_gain / avg_loss
            df['RSI'] = 100.0 - (100.0 / (1.0 + relative_strength))
            
            # Quick ATR calculation
            df['ATR'] = np.maximum(
                df['High'] - df['Low'],
                np.abs(df['High'] - df['Close'].shift(1)),
                np.abs(df['Low'] - df['Close'].shift(1))
            ).rolling(window=7).mean()  # Shorter period for intraday
            
            # Enhanced trend detection
            df['Trend_Strength'] = ((df['Close'] - df['EMA_20']) / df['EMA_20']) * 100
            df['Short_Trend'] = ((df['Close'] - df['EMA_5']) / df['EMA_5']) * 100
            
            # Volume analysis
            df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            return df
            
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return None


    def analyze_stock(self, symbol):
        """Enhanced stock analysis with focus on real-time intraday signals"""
        try:
            # Get 1-minute data for intraday
            df_intraday = self.get_stock_data(symbol, duration='1d', interval='1m')
            # Get 5-minute data for swing and momentum
            df_daily = self.get_stock_data(symbol, duration='5d', interval='5m')
            
            if df_intraday is None or df_daily is None:
                return None
                
            # Calculate indicators
            df_intraday = self.calculate_indicators(df_intraday)
            df_daily = self.calculate_indicators(df_daily)
            
            if df_intraday is None or df_daily is None:
                return None
                
            current_close = df_intraday['Close'].iloc[-1]
            last_update = df_intraday.index[-1]
            
            # Cache check for intraday signals
            signal_key = f"{symbol}_{last_update}"
            if signal_key in self.intraday_cache:
                return self.intraday_cache[signal_key]
            
            signals = {
                'intraday': self.get_intraday_signals(df_intraday),
                'swing': self.get_swing_signals(df_daily),
                'momentum': self.get_momentum_signals(df_daily)
            }
            
            result = {
                'symbol': symbol,
                'close': current_close,
                'volume': df_intraday['Volume'].iloc[-1],
                'signals': signals,
                'last_update': last_update
            }
            
            # Cache intraday result
            self.intraday_cache[signal_key] = result
            
            # Clean old cache entries
            self.clean_intraday_cache()
            
            return result
            
        except Exception as e:
            logger.info(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def clean_intraday_cache(self):
        """Clean old cache entries"""
        current_time = datetime.now(self.ist_tz)
        expired_keys = [
            k for k in self.intraday_cache.keys()
            if current_time - self.intraday_cache[k]['last_update'].tz_localize(self.ist_tz) > timedelta(minutes=5)
        ]
        for k in expired_keys:
            del self.intraday_cache[k]

    def get_intraday_signals(self, df):
        """Enhanced intraday signal detection"""
        try:
            current_close = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2]
            ema_5 = df['EMA_5'].iloc[-1]
            ema_10 = df['EMA_10'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            short_trend = df['Short_Trend'].iloc[-1]
            
            signal_strength = 0
            reasons = []
            
            # Quick momentum check
            if current_close > ema_5 and prev_close <= df['EMA_5'].iloc[-2]:
                signal_strength += 2
                reasons.append("Price crossed above 5 EMA")
            
            # Trend confirmation
            if ema_5 > ema_10:
                signal_strength += 1
                reasons.append("Short-term trend up")
            
            # RSI momentum
            if 45 <= rsi <= 65:
                signal_strength += 1
                reasons.append("RSI in momentum zone")
            
            # Volume spike
            if volume_ratio > 1.5:
                signal_strength += 1
                reasons.append("Volume spike detected")
            
            # Short-term trend strength
            if short_trend > 0.5:
                signal_strength += 1
                reasons.append("Strong short-term momentum")
            
            # Calculate tight stops for intraday
            atr = df['ATR'].iloc[-1]
            suggested_sl = current_close - (atr * 0.75)  # Tighter stop for intraday
            suggested_target = current_close + (atr * 1.5)  # 1:2 risk-reward
            
            return {
                'signal': 'buy' if signal_strength >= 3 else 'neutral',
                'strength': signal_strength,
                'reasons': reasons,
                'suggested_entry': current_close,
                'suggested_sl': suggested_sl,
                'suggested_target': suggested_target,
                'timestamp': df.index[-1]
            }
            
        except Exception as e:
            logger.info(f"Error in intraday signals: {str(e)}")
            return None

    def get_swing_signals(self, df):
        """
        Swing Strategy:
        - Pullback to 10 EMA
        - Low RSI (30-45)
        - Overall uptrend
        """
        try:
            current_close = df['Close'].iloc[-1]
            ema_10 = df['EMA_10'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            
            signal_strength = 0
            reasons = []
            
            # Check for pullback to 10 EMA
            price_to_ema = abs((current_close - ema_10) / ema_10) * 100
            if price_to_ema <= 0.5:  # Within 0.5% of 10 EMA
                signal_strength += 1
                reasons.append("Price near 10 EMA")
            
            # Check RSI condition (looking for low RSI)
            if 30 <= rsi <= 45:
                signal_strength += 1
                reasons.append("RSI indicates pullback")
                
            # Trend confirmation
            if df['Trend_Strength'].iloc[-1] > 0:
                signal_strength += 1
                reasons.append("Overall uptrend intact")
                
            # Calculate entry/exit points
            atr = df['ATR'].iloc[-1]
            suggested_sl = current_close - (atr * 1.5)
            suggested_target = current_close + (atr * 3)  # 1:2 risk-reward
            
            return {
                'signal': 'buy' if signal_strength >= 2 else 'neutral',
                'strength': signal_strength,
                'reasons': reasons,
                'suggested_entry': current_close,
                'suggested_sl': suggested_sl,
                'suggested_target': suggested_target
            }
            
        except Exception as e:
            logger.info(f"Error in swing signals: {str(e)}")
            return None

    def get_momentum_signals(self, df):
        """
        Momentum Strategy:
        - 10 EMA crossover
        - RSI > 60
        - Strong volume and trend
        """
        try:
            current_close = df['Close'].iloc[-1]
            ema_10 = df['EMA_10'].iloc[-1]
            ema_20 = df['EMA_20'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            
            signal_strength = 0
            reasons = []
            
            # Check for 10 EMA crossover
            if (ema_10 > ema_20 and 
                df['EMA_10'].iloc[-2] <= df['EMA_20'].iloc[-2]):
                signal_strength += 2
                reasons.append("10 EMA crossed above 20 EMA")
            
            # Strong RSI
            if rsi > 60:
                signal_strength += 1
                reasons.append("RSI > 60")
                
            # Volume confirmation
            if volume_ratio > 1.5:
                signal_strength += 1
                reasons.append("Strong volume confirmation")
                
            # Trend strength
            if df['Trend_Strength'].iloc[-1] > 2:
                signal_strength += 1
                reasons.append("Strong uptrend")
                
            # Calculate entry/exit points
            atr = df['ATR'].iloc[-1]
            suggested_sl = current_close - (atr * 2)
            suggested_target = current_close + (atr * 4)  # 1:2 risk-reward
            
            return {
                'signal': 'buy' if signal_strength >= 3 else 'neutral',
                'strength': signal_strength,
                'reasons': reasons,
                'suggested_entry': current_close,
                'suggested_sl': suggested_sl,
                'suggested_target': suggested_target
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
        """Send alert to Telegram with rate limiting"""
        try:
            await asyncio.sleep(1)  # Basic rate limiting
            await self.bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

    def format_signal_message(self, stock, strategy):
        """Format the signal message for Telegram"""
        message = f"🔔 <b>{strategy.upper()} Trading Signal</b>\n\n"
        message += f"Stock: {stock['symbol']}\n"
        message += f"Current Price: ₹{stock['close']:.2f}\n"
        message += f"Signal Strength: {stock['signal']['strength']}\n"
        message += f"Entry: ₹{stock['signal']['suggested_entry']:.2f}\n"
        message += f"Stop Loss: ₹{stock['signal']['suggested_sl']:.2f}\n"
        message += f"Target: ₹{stock['signal']['suggested_target']:.2f}\n"
        message += f"Reasons:\n{chr(8226)} " + f"\n{chr(8226)} ".join(stock['signal']['reasons'])
        return message

    def scan_market(self):
        """Scan all stocks and generate trading opportunities"""
        self.get_nifty_stocks()
          
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
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
                        'volume': result['volume']
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
        """
        Check if there are significant changes in the signal
        """
        key = f"{stock}_{strategy}"
        last_signal = self.last_signals.get(key, {})
        
        if not last_signal:
            return True
            
        # Check for meaningful changes
        significant_changes = [
            abs(new_signal['suggested_entry'] - last_signal.get('suggested_entry', 0)) > 0.5,  # 0.5% price change
            abs(new_signal['strength'] - last_signal.get('strength', 0)) >= 1,  # Change in signal strength
            set(new_signal['reasons']) != set(last_signal.get('reasons', [])),  # New reasons
            new_signal['signal'] != last_signal.get('signal', 'neutral')  # Change in signal type
        ]
        
        return any(significant_changes)

    async def scan_and_alert(self):
            """Enhanced scanning with focus on real-time intraday alerts"""
            if not self.is_market_open():
                logger.info("Market is closed")
                return

            opportunities = self.scan_market()
            new_signals = {}
            alerts_sent = False
            
            # Prioritize intraday signals
            if opportunities['intraday']:
                for stock in opportunities['intraday']:
                    key = f"{stock['symbol']}_intraday"
                    
                    # Check if signal is new or significantly changed
                    if self.has_significant_changes(stock['symbol'], 'intraday', stock['signal']):
                        message = self.format_signal_message(stock, 'intraday')
                        await self.send_telegram_alert(message)
                        alerts_sent = True
                        
                    new_signals[key] = stock['signal']
            
            # Process other timeframes
            for strategy in ['swing', 'momentum']:
                for stock in opportunities[strategy]:
                    key = f"{stock['symbol']}_{strategy}"
                    if self.has_significant_changes(stock['symbol'], strategy, stock['signal']):
                        message = self.format_signal_message(stock, strategy)
                        await self.send_telegram_alert(message)
                        alerts_sent = True
                    new_signals[key] = stock['signal']
            
            if alerts_sent:
                summary = f"✨ Scan completed at {datetime.now(self.ist_tz).strftime('%H:%M:%S')}"
                await self.send_telegram_alert(summary)
            
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