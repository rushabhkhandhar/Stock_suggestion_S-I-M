import yfinance as yf
import pandas as pd
import numpy as np

import telegram
import asyncio
import pytz
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
# from dotenv import load_dotenv
import os
import sys

import logging
# load_dotenv()
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
        self.screened_signals = set()  # To avoid duplicate alerts
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        
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
        """Custom indicator calculations"""
        try:
            # Exponential Moving Averages
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            # Custom RSI Calculation
            delta = df['Close'].diff()
            
            gains = delta.clip(lower=0)
            losses = -delta.clip(upper=0)
            
            avg_gain = gains.rolling(window=14).mean()
            avg_loss = losses.rolling(window=14).mean()
            
            relative_strength = avg_gain / avg_loss
            df['RSI'] = 100.0 - (100.0 / (1.0 + relative_strength))
            
            # ATR Approximation
            df['ATR'] = np.maximum(
                df['High'] - df['Low'], 
                np.abs(df['High'] - df['Close'].shift(1)),
                np.abs(df['Low'] - df['Close'].shift(1))
            ).rolling(window=14).mean()
            
            # Trend Strength
            df['Trend_Strength'] = ((df['Close'] - df['EMA_20']) / df['EMA_20']) * 100
            
            # Volume Ratio
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            return df
        
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return None

    def analyze_stock(self, symbol):
        """Analyze a single stock for trading opportunities"""
        try:
            # Get intraday data (5-minute intervals for last 5 days)
            df_intraday = self.get_stock_data(symbol, duration='5d', interval='5m')
            # Get daily data for swing and momentum (1 month of daily data)
            df_daily = self.get_stock_data(symbol, duration='1mo', interval='1d')
            
            if df_intraday is None or df_daily is None:
                return None
                
            # Calculate indicators for both timeframes
            df_intraday = self.calculate_indicators(df_intraday)
            df_daily = self.calculate_indicators(df_daily)
            
            if df_intraday is None or df_daily is None:
                return None
                
            # Get latest values
            current_close = df_intraday['Close'].iloc[-1]
            
            signals = {
                'intraday': self.get_intraday_signals(df_intraday),
                'swing': self.get_swing_signals(df_daily),
                'momentum': self.get_momentum_signals(df_daily)
            }
            
            return {
                'symbol': symbol,
                'close': current_close,
                'volume': df_intraday['Volume'].iloc[-1],
                'signals': signals,
            }
            
        except Exception as e:
            logger.info(f"Error analyzing {symbol}: {str(e)}")
            return None

    def get_intraday_signals(self, df):
        """
        Intraday Strategy:
        - Breakout above 20 EMA
        - RSI > 50
        - Volume confirmation
        """
        try:
            current_close = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2]
            ema_20 = df['EMA_20'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            
            signal_strength = 0
            reasons = []
            
            # Check for breakout above 20 EMA
            if current_close > ema_20 and prev_close <= df['EMA_20'].iloc[-2]:
                signal_strength += 1
                reasons.append("Breakout above 20 EMA")
            
            # Check RSI condition
            if rsi > 50:
                signal_strength += 1
                reasons.append("RSI > 50")
            
            # Volume confirmation
            if volume_ratio > 1.2:
                signal_strength += 1
                reasons.append("High volume confirmation")
                
            # Calculate entry/exit points
            atr = df['ATR'].iloc[-1]
            suggested_sl = current_close - atr
            suggested_target = current_close + (atr * 2)  # 1:2 risk-reward
            
            return {
                'signal': 'buy' if signal_strength >= 2 else 'neutral',
                'strength': signal_strength,
                'reasons': reasons,
                'suggested_entry': current_close,
                'suggested_sl': suggested_sl,
                'suggested_target': suggested_target
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

    async def scan_and_alert(self):
        """Scan market and send alerts for new signals"""
        if not self.is_market_open():
            logger.info("Market is closed")
            return

        opportunities = self.scan_market()
        
        for strategy, stocks in opportunities.items():
            for stock in stocks:
                # Create a unique identifier for this signal
                signal_id = f"{stock['symbol']}_{strategy}_{datetime.now(self.ist_tz).strftime('%Y%m%d')}"
                
                # Only alert if we haven't seen this signal today
                if signal_id not in self.screened_signals:
                    message = self.format_signal_message(stock, strategy)
                    await self.send_telegram_alert(message)
                    self.screened_signals.add(signal_id)
                    
                    # Clear old signals (older than 24 hours)
                    current_time = datetime.now(self.ist_tz)
                    self.screened_signals = {
                        signal for signal in self.screened_signals
                        if signal.split('_')[2] == current_time.strftime('%Y%m%d')
                    }

async def main():
    """Main async function with 5-minute scheduling"""
    try:
        # Get tokens from environment variables
        TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
        TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
        
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            logger.error("Telegram credentials not found!")
            return
        
        # Initialize screener
        screener = StockScreener(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        
        # Send initial status message
        await screener.send_telegram_alert("🟢 Stock Screener Started\nMonitoring market for trading opportunities...")
        
        while True:
            # Check if market is open
            if not screener.is_market_open():
                logger.info("Market is closed. Waiting for next market day.")
                await screener.send_telegram_alert("❌ Market is closed. Will resume on next market day.")
                # Wait for 1 hour before checking again
                await asyncio.sleep(3600)
                continue
            
            try:
                # Perform market scan and send alerts
                await screener.scan_and_alert()
                logger.info("Market scan completed. Waiting for 5 minutes...")
                
                # Wait for 5 minutes before next scan
                await asyncio.sleep(300)  # 300 seconds = 5 minutes
            
            except Exception as e:
                error_message = f"❌ Error during market scan: {str(e)}"
                logger.error(error_message)
                await screener.send_telegram_alert(error_message)
                # Wait for 1 minute before retrying after error
                await asyncio.sleep(60)
    
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