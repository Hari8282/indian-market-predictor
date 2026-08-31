"""
Indian Stock Market Predictor - Multi-Timeframe Backend
Real-time data with multiple timeframe support
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import logging
from functools import lru_cache
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# curl_cffi is used to impersonate a real browser's TLS fingerprint so Yahoo
# Finance doesn't block requests coming from cloud/datacenter IPs (Render,
# Railway, etc). It's optional at import time — if it's not installed for
# any reason, we fall back to plain yfinance requests instead of crashing.
try:
    from curl_cffi import requests as curl_requests
    CURL_CFFI_AVAILABLE = True
    logger.info("curl_cffi loaded successfully - using browser-impersonated sessions")
except ImportError as e:
    CURL_CFFI_AVAILABLE = False
    logger.warning(f"curl_cffi not available ({e}) - falling back to plain yfinance requests. "
                    f"This may cause Yahoo Finance to block requests from cloud IPs.")

app = Flask(__name__)

# CORS Configuration - Allow requests from GitHub Pages and local development
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Cache configuration
CACHE_TIMEOUT = 60  # seconds

# Shared browser-impersonating session for all Yahoo Finance requests.
def get_yf_session():
    """Create a fresh impersonated session for Yahoo Finance requests, if available"""
    if not CURL_CFFI_AVAILABLE:
        return None
    try:
        return curl_requests.Session(impersonate="chrome")
    except Exception as e:
        logger.warning(f"Could not create curl_cffi session, falling back to default: {e}")
        return None

_YF_SESSION = get_yf_session()

def get_ticker(symbol):
    """Get a yfinance Ticker using the browser-impersonating session when available"""
    global _YF_SESSION
    try:
        if _YF_SESSION is not None:
            return yf.Ticker(symbol, session=_YF_SESSION)
        return yf.Ticker(symbol)
    except Exception as e:
        logger.warning(f"Ticker creation with session failed for {symbol}, retrying without session: {e}")
        return yf.Ticker(symbol)

# Global market indices
GLOBAL_INDICES = {
    'asian': {
        '^N225': 'Nikkei 225',
        '^HSI': 'Hang Seng',
        '000001.SS': 'Shanghai Composite',
        '^KS11': 'KOSPI'
    },
    'european': {
        '^FTSE': 'FTSE 100',
        '^GDAXI': 'DAX',
        '^FCHI': 'CAC 40'
    },
    'us': {
        '^GSPC': 'S&P 500',
        '^IXIC': 'NASDAQ',
        '^DJI': 'Dow Jones'
    }
}

# Timeframe configurations with CPR period mapping
TIMEFRAMES = {
    '1m': {'period': '1d', 'interval': '1m', 'label': '1 Minute', 'cpr_basis': 'daily'},
    '5m': {'period': '5d', 'interval': '5m', 'label': '5 Minutes', 'cpr_basis': 'daily'},
    '15m': {'period': '5d', 'interval': '15m', 'label': '15 Minutes', 'cpr_basis': 'daily'},
    '30m': {'period': '5d', 'interval': '30m', 'label': '30 Minutes', 'cpr_basis': 'daily'},
    '1h': {'period': '1mo', 'interval': '1h', 'label': '1 Hour', 'cpr_basis': 'weekly'},
    '1d': {'period': '6mo', 'interval': '1d', 'label': '1 Day', 'cpr_basis': 'weekly'},
    '1wk': {'period': '2y', 'interval': '1wk', 'label': '1 Week', 'cpr_basis': 'monthly'}
}

def get_cpr_period_data(symbol, timeframe):
    """
    Fetch previous period data for CPR calculation based on timeframe
    
    - 1m, 5m, 15m, 30m: Previous day's OHLC (Daily CPR)
    - 1h, 1d: Previous week's OHLC (Weekly CPR)
    - 1wk: Previous month's OHLC (Monthly CPR)
    
    Returns None on any failure so caller can gracefully fall back.
    """
    try:
        ticker = get_ticker(symbol)
        cpr_basis = TIMEFRAMES.get(timeframe, {}).get('cpr_basis', 'daily')
        
        if cpr_basis == 'daily':
            data = ticker.history(period='5d', interval='1d', timeout=10)
            if data is not None and len(data) >= 2:
                prev_day = data.iloc[-2]
                return {
                    'high': float(prev_day['High']),
                    'low': float(prev_day['Low']),
                    'close': float(prev_day['Close']),
                    'date': data.index[-2].strftime('%Y-%m-%d'),
                    'basis': 'Daily',
                    'period_label': f"Previous Day ({data.index[-2].strftime('%d %b %Y')})"
                }
        
        elif cpr_basis == 'weekly':
            data = ticker.history(period='1mo', interval='1wk', timeout=10)
            if data is not None and len(data) >= 2:
                prev_week = data.iloc[-2]
                return {
                    'high': float(prev_week['High']),
                    'low': float(prev_week['Low']),
                    'close': float(prev_week['Close']),
                    'date': data.index[-2].strftime('%Y-%m-%d'),
                    'basis': 'Weekly',
                    'period_label': f"Previous Week ({data.index[-2].strftime('%d %b %Y')})"
                }
        
        elif cpr_basis == 'monthly':
            data = ticker.history(period='1y', interval='1mo', timeout=10)
            if data is not None and len(data) >= 2:
                prev_month = data.iloc[-2]
                return {
                    'high': float(prev_month['High']),
                    'low': float(prev_month['Low']),
                    'close': float(prev_month['Close']),
                    'date': data.index[-2].strftime('%Y-%m-%d'),
                    'basis': 'Monthly',
                    'period_label': f"Previous Month ({data.index[-2].strftime('%b %Y')})"
                }
        
        logger.warning(f"CPR period data insufficient for {symbol} ({cpr_basis})")
        return None
        
    except Exception as e:
        logger.warning(f"CPR period fetch failed for {symbol} (non-fatal, will fall back): {e}")
        return None

def calculate_cpr_with_period(symbol, timeframe, current_data):
    """
    Calculate CPR based on appropriate previous period
    Returns CPR values along with the basis period information
    """
    period_data = get_cpr_period_data(symbol, timeframe)
    
    if period_data is None:
        # Fallback to current data if period data unavailable
        if current_data is None or len(current_data) == 0:
            return {
                'pivot': 0.0, 'tc': 0.0, 'bc': 0.0,
                'basis': 'N/A', 'period_label': 'N/A', 'date': 'N/A'
            }
        
        high = float(current_data['High'].iloc[-1])
        low = float(current_data['Low'].iloc[-1])
        close = float(current_data['Close'].iloc[-1])
        basis = 'Current'
        period_label = 'Current Period'
        date = 'N/A'
    else:
        high = period_data['high']
        low = period_data['low']
        close = period_data['close']
        basis = period_data['basis']
        period_label = period_data['period_label']
        date = period_data['date']
    
    # CPR Calculation
    pivot = (high + low + close) / 3
    bc = (high + low) / 2
    tc = (pivot - bc) + pivot
    
    return {
        'pivot': float(round(pivot, 2)),
        'tc': float(round(tc, 2)),
        'bc': float(round(bc, 2)),
        'high': float(round(high, 2)),
        'low': float(round(low, 2)),
        'close': float(round(close, 2)),
        'basis': str(basis),
        'period_label': str(period_label),
        'date': str(date)
    }

def calculate_support_resistance_with_period(symbol, timeframe, current_data):
    """
    Calculate Support and Resistance based on appropriate previous period
    """
    period_data = get_cpr_period_data(symbol, timeframe)
    
    if period_data is None:
        # Fallback to current data
        if current_data is None or len(current_data) == 0:
            return [], [], {'basis': 'N/A', 'period_label': 'N/A'}
        
        high = float(current_data['High'].iloc[-1])
        low = float(current_data['Low'].iloc[-1])
        close = float(current_data['Close'].iloc[-1])
        basis = 'Current'
        period_label = 'Current Period'
    else:
        high = period_data['high']
        low = period_data['low']
        close = period_data['close']
        basis = period_data['basis']
        period_label = period_data['period_label']
    
    # Pivot Point
    pivot = (high + low + close) / 3
    
    # Standard Pivot Point Levels
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    r4 = high + 3 * (pivot - low)
    
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    s4 = low - 3 * (high - pivot)
    
    # Camarilla Pivot Levels (additional)
    range_hl = high - low
    cam_r1 = close + (range_hl * 1.1 / 12)
    cam_r2 = close + (range_hl * 1.1 / 6)
    cam_r3 = close + (range_hl * 1.1 / 4)
    cam_r4 = close + (range_hl * 1.1 / 2)
    
    cam_s1 = close - (range_hl * 1.1 / 12)
    cam_s2 = close - (range_hl * 1.1 / 6)
    cam_s3 = close - (range_hl * 1.1 / 4)
    cam_s4 = close - (range_hl * 1.1 / 2)
    
    resistance = [
        {'level': 'R1', 'value': float(round(r1, 2)), 'type': 'Standard'},
        {'level': 'R2', 'value': float(round(r2, 2)), 'type': 'Standard'},
        {'level': 'R3', 'value': float(round(r3, 2)), 'type': 'Standard'},
        {'level': 'R4', 'value': float(round(r4, 2)), 'type': 'Standard'}
    ]
    
    support = [
        {'level': 'S1', 'value': float(round(s1, 2)), 'type': 'Standard'},
        {'level': 'S2', 'value': float(round(s2, 2)), 'type': 'Standard'},
        {'level': 'S3', 'value': float(round(s3, 2)), 'type': 'Standard'},
        {'level': 'S4', 'value': float(round(s4, 2)), 'type': 'Standard'}
    ]
    
    sr_info = {
        'basis': str(basis),
        'period_label': str(period_label)
    }
    
    return support, resistance, sr_info

def fetch_market_data(symbol, timeframe='15m'):
    """
    Fetch market data for specific timeframe.
    Retries once with a fresh session if the first attempt fails, since
    Yahoo Finance occasionally rejects a request even with impersonation.
    """
    if timeframe not in TIMEFRAMES:
        timeframe = '15m'
    
    config = TIMEFRAMES[timeframe]
    
    for attempt in range(2):
        try:
            ticker = get_ticker(symbol)
            data = ticker.history(period=config['period'], interval=config['interval'], timeout=15)
            
            if data is None or data.empty:
                logger.warning(f"Attempt {attempt+1}: empty response for {symbol} on {timeframe}")
                if attempt == 0:
                    global _YF_SESSION
                    _YF_SESSION = get_yf_session()  # fresh session before retry
                    time.sleep(1)
                    continue
                return None
            
            logger.info(f"Fetched {len(data)} candles for {symbol} on {timeframe}")
            return data
        except Exception as e:
            logger.error(f"Attempt {attempt+1} error fetching {symbol} on {timeframe}: {type(e).__name__}: {e}")
            if attempt == 0:
                _YF_SESSION = get_yf_session()
                time.sleep(1)
                continue
            return None
    
    return None

def calculate_cpr(data):
    """Calculate Central Pivot Range"""
    if data is None or len(data) == 0:
        return {'pivot': 0.0, 'tc': 0.0, 'bc': 0.0}
    
    high = float(data['High'].iloc[-1])
    low = float(data['Low'].iloc[-1])
    close = float(data['Close'].iloc[-1])
    
    pivot = (high + low + close) / 3
    bc = (high + low) / 2
    tc = (pivot - bc) + pivot
    
    return {
        'pivot': float(round(pivot, 2)),
        'tc': float(round(tc, 2)),
        'bc': float(round(bc, 2))
    }

def calculate_support_resistance(data, num_levels=4):
    """Calculate support and resistance levels"""
    if data is None or len(data) == 0:
        return [], []
    
    high = float(data['High'].iloc[-1])
    low = float(data['Low'].iloc[-1])
    close = float(data['Close'].iloc[-1])
    
    pivot = (high + low + close) / 3
    
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    r4 = high + 3 * (pivot - low)
    
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    s4 = low - 3 * (high - pivot)
    
    resistance = [float(round(r1, 2)), float(round(r2, 2)), float(round(r3, 2)), float(round(r4, 2))]
    support = [float(round(s1, 2)), float(round(s2, 2)), float(round(s3, 2)), float(round(s4, 2))]
    
    return support, resistance

def calculate_volume_analysis(data):
    """Calculate volume-based indicators"""
    if data is None or len(data) < 20:
        return {
            'current_volume': 0,
            'avg_volume': 0,
            'volume_ratio': 1.0,
            'volume_trend': 'neutral'
        }
    
    try:
        current_volume = float(data['Volume'].iloc[-1])
        avg_volume = float(data['Volume'].tail(20).mean())
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        volume_trend = 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.5 else 'normal'
        
        return {
            'current_volume': int(round(current_volume, 0)),
            'avg_volume': int(round(avg_volume, 0)),
            'volume_ratio': float(round(volume_ratio, 2)),
            'volume_trend': str(volume_trend)
        }
    except Exception as e:
        logger.warning(f"Volume calculation error: {e}")
        return {
            'current_volume': 0,
            'avg_volume': 0,
            'volume_ratio': 1.0,
            'volume_trend': 'neutral'
        }

def generate_candlestick_data(data, max_candles=100):
    """Generate candlestick data"""
    if data is None or len(data) == 0:
        return []
    
    candles = []
    recent_data = data.tail(max_candles)
    
    for idx, (timestamp, row) in enumerate(recent_data.iterrows()):
        candles.append({
            'time': int(idx),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M'),
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'close': round(float(row['Close']), 2),
            'volume': int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0,
            'isBullish': bool(row['Close'] > row['Open'])
        })
    
    return candles

def calculate_technical_indicators(data):
    """Calculate multiple technical indicators"""
    if data is None or len(data) < 20:
        return {}
    
    try:
        indicators = {}
        
        # RSI
        if len(data) >= 14:
            rsi_indicator = ta.momentum.RSIIndicator(data['Close'], window=14)
            rsi_value = rsi_indicator.rsi().iloc[-1]
            if not pd.isna(rsi_value):
                indicators['rsi'] = float(round(rsi_value, 2))
        
        # Moving Averages
        if len(data) >= 20:
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            if not pd.isna(sma_20):
                indicators['sma_20'] = float(round(sma_20, 2))
                
        if len(data) >= 50:
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            if not pd.isna(sma_50):
                indicators['sma_50'] = float(round(sma_50, 2))
                
        if len(data) >= 200:
            sma_200 = data['Close'].rolling(window=200).mean().iloc[-1]
            if not pd.isna(sma_200):
                indicators['sma_200'] = float(round(sma_200, 2))
        
        # EMA
        if len(data) >= 12:
            ema_12 = data['Close'].ewm(span=12, adjust=False).mean().iloc[-1]
            if not pd.isna(ema_12):
                indicators['ema_12'] = float(round(ema_12, 2))
                
        if len(data) >= 26:
            ema_26 = data['Close'].ewm(span=26, adjust=False).mean().iloc[-1]
            if not pd.isna(ema_26):
                indicators['ema_26'] = float(round(ema_26, 2))
        
        # MACD
        if len(data) >= 26:
            macd_indicator = ta.trend.MACD(data['Close'])
            macd_val = macd_indicator.macd().iloc[-1]
            macd_sig = macd_indicator.macd_signal().iloc[-1]
            macd_diff = macd_indicator.macd_diff().iloc[-1]
            
            if not pd.isna(macd_val):
                indicators['macd'] = float(round(macd_val, 2))
            if not pd.isna(macd_sig):
                indicators['macd_signal'] = float(round(macd_sig, 2))
            if not pd.isna(macd_diff):
                indicators['macd_diff'] = float(round(macd_diff, 2))
        
        # Bollinger Bands
        if len(data) >= 20:
            bb_indicator = ta.volatility.BollingerBands(data['Close'], window=20)
            bb_upper = bb_indicator.bollinger_hband().iloc[-1]
            bb_middle = bb_indicator.bollinger_mavg().iloc[-1]
            bb_lower = bb_indicator.bollinger_lband().iloc[-1]
            
            if not pd.isna(bb_upper):
                indicators['bb_upper'] = float(round(bb_upper, 2))
            if not pd.isna(bb_middle):
                indicators['bb_middle'] = float(round(bb_middle, 2))
            if not pd.isna(bb_lower):
                indicators['bb_lower'] = float(round(bb_lower, 2))
        
        # ATR (Average True Range)
        if len(data) >= 14:
            atr_indicator = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14)
            atr_val = atr_indicator.average_true_range().iloc[-1]
            if not pd.isna(atr_val):
                indicators['atr'] = float(round(atr_val, 2))
        
        # Momentum
        if len(data) >= 10:
            current_price = float(data['Close'].iloc[-1])
            old_price = float(data['Close'].iloc[-10])
            momentum = ((current_price - old_price) / old_price) * 100
            indicators['momentum'] = float(round(momentum, 2))
        
        return indicators
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}

def predict_market_direction(nifty_data, global_markets, indicators):
    """Enhanced prediction with multiple factors"""
    if nifty_data is None or len(nifty_data) < 20:
        return {
            'direction': 'neutral',
            'confidence': 50.0,
            'sentiment': 'neutral',
            'signals': {},
            'global_positive_ratio': 50.0
        }
    
    bullish_signals = 0
    total_signals = 0
    signals = {}
    
    # RSI Signal
    if 'rsi' in indicators:
        rsi = indicators['rsi']
        if rsi < 30:
            bullish_signals += 1
            signals['rsi'] = 'bullish'
        elif rsi > 70:
            signals['rsi'] = 'bearish'
        else:
            bullish_signals += 0.5
            signals['rsi'] = 'neutral'
        total_signals += 1
    
    # Moving Average Signal
    current_price = float(nifty_data['Close'].iloc[-1])
    if 'sma_20' in indicators:
        if current_price > indicators['sma_20']:
            bullish_signals += 1
            signals['sma'] = 'bullish'
        else:
            signals['sma'] = 'bearish'
        total_signals += 1
    
    # MACD Signal
    if 'macd' in indicators and 'macd_signal' in indicators:
        if indicators['macd'] > indicators['macd_signal']:
            bullish_signals += 1
            signals['macd'] = 'bullish'
        else:
            signals['macd'] = 'bearish'
        total_signals += 1
    
    # Momentum Signal
    if 'momentum' in indicators:
        if indicators['momentum'] > 1:
            bullish_signals += 1
            signals['momentum'] = 'bullish'
        elif indicators['momentum'] < -1:
            signals['momentum'] = 'bearish'
        else:
            bullish_signals += 0.5
            signals['momentum'] = 'neutral'
        total_signals += 1
    
    # Global Market Sentiment
    positive_global = sum(1 for region in global_markets.values() 
                         for market in region if market.get('change', 0) > 0)
    total_global = sum(len(region) for region in global_markets.values())
    global_ratio = positive_global / total_global if total_global > 0 else 0.5
    
    if global_ratio > 0.6:
        bullish_signals += 1
        signals['global'] = 'bullish'
    elif global_ratio < 0.4:
        signals['global'] = 'bearish'
    else:
        bullish_signals += 0.5
        signals['global'] = 'neutral'
    total_signals += 1
    
    # Calculate confidence
    confidence = (bullish_signals / total_signals) * 100 if total_signals > 0 else 50
    
    # Determine direction
    if confidence > 65:
        direction = 'bullish'
        sentiment = 'positive'
    elif confidence < 35:
        direction = 'bearish'
        sentiment = 'negative'
    else:
        direction = 'neutral'
        sentiment = 'neutral'
    
    return {
        'direction': str(direction),
        'confidence': float(round(confidence, 1)),
        'sentiment': str(sentiment),
        'signals': signals,
        'global_positive_ratio': float(round(global_ratio * 100, 1))
    }

def process_market_data(symbol, timeframe='15m'):
    """Process complete market data for a symbol with period-based CPR"""
    try:
        data = fetch_market_data(symbol, timeframe)
        
        if data is None or len(data) == 0:
            logger.warning(f"No data for {symbol} on {timeframe}")
            return None
        
        current_price = float(data['Close'].iloc[-1])
        prev_close = float(data['Open'].iloc[0])
        
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
        
        # Calculate CPR with appropriate period basis
        cpr = calculate_cpr_with_period(symbol, timeframe, data)
        
        # Calculate Support/Resistance with appropriate period basis
        support, resistance, sr_info = calculate_support_resistance_with_period(symbol, timeframe, data)
        
        candles = generate_candlestick_data(data)
        volume = calculate_volume_analysis(data)
        indicators = calculate_technical_indicators(data)
        
        # High/Low for the period
        high_price = float(data['High'].max())
        low_price = float(data['Low'].min())
        
        # Get CPR basis info
        cpr_basis = TIMEFRAMES.get(timeframe, {}).get('cpr_basis', 'daily')
        
        return {
            'current': float(round(current_price, 2)),
            'open': float(round(float(data['Open'].iloc[0]), 2)),
            'high': float(round(high_price, 2)),
            'low': float(round(low_price, 2)),
            'change': float(round(change, 2)),
            'changePercent': float(round(change_percent, 2)),
            'cpr': cpr,
            'support': support,
            'resistance': resistance,
            'sr_info': sr_info,
            'cpr_basis': str(cpr_basis),
            'candleData': candles,
            'volume': volume,
            'indicators': indicators,
            'dataPoints': int(len(data))
        }
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None

def fetch_global_markets():
    """Fetch global market data"""
    global_markets = {}
    
    for region, indices in GLOBAL_INDICES.items():
        global_markets[region] = []
        for symbol, name in indices.items():
            try:
                data = fetch_market_data(symbol, '1d')
                if data is not None and len(data) > 0:
                    current = float(data['Close'].iloc[-1])
                    prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else current
                    change = ((current - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                    
                    global_markets[region].append({
                        'name': str(name),
                        'value': float(round(current, 2)),
                        'change': float(round(change, 2)),
                        'trend': 'up' if change > 0 else 'down'
                    })
            except Exception as e:
                logger.warning(f"Skipping {name}: {e}")
                continue
    
    return global_markets

@app.route('/api/debug', methods=['GET'])
def debug_yfinance():
    """
    Diagnostic endpoint to pinpoint exactly why Yahoo Finance fetches fail.
    Safe to expose - no sensitive data, just connectivity diagnostics.
    """
    result = {
        'curl_cffi_available': CURL_CFFI_AVAILABLE,
        'session_created': _YF_SESSION is not None,
        'yfinance_version': getattr(yf, '__version__', 'unknown')
    }
    
    # Test 1: Try fetching Nifty (the symbol that's failing)
    try:
        ticker = get_ticker('^NSEI')
        data = ticker.history(period='5d', interval='15m', timeout=15)
        result['nifty_fetch_success'] = data is not None and not data.empty
        result['nifty_rows'] = int(len(data)) if data is not None else 0
    except Exception as e:
        result['nifty_fetch_success'] = False
        result['nifty_error_type'] = type(e).__name__
        result['nifty_error_message'] = str(e)
    
    # Test 2: Try a well-known US ticker to isolate whether it's Yahoo-wide
    # blocking or specific to NSE/Indian symbols
    try:
        ticker2 = get_ticker('AAPL')
        data2 = ticker2.history(period='5d', interval='1d', timeout=15)
        result['aapl_fetch_success'] = data2 is not None and not data2.empty
        result['aapl_rows'] = int(len(data2)) if data2 is not None else 0
    except Exception as e:
        result['aapl_fetch_success'] = False
        result['aapl_error_type'] = type(e).__name__
        result['aapl_error_message'] = str(e)
    
    # Test 3: Raw HTTP check to Yahoo's endpoint directly (bypassing yfinance)
    try:
        if CURL_CFFI_AVAILABLE:
            test_session = curl_requests.Session(impersonate="chrome")
            resp = test_session.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/AAPL",
                timeout=15
            )
            result['raw_http_status'] = resp.status_code
            result['raw_http_success'] = resp.status_code == 200
        else:
            import urllib.request
            req = urllib.request.Request(
                "https://query1.finance.yahoo.com/v8/finance/chart/AAPL",
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            resp = urllib.request.urlopen(req, timeout=15)
            result['raw_http_status'] = resp.status
            result['raw_http_success'] = resp.status == 200
    except Exception as e:
        result['raw_http_success'] = False
        result['raw_http_error_type'] = type(e).__name__
        result['raw_http_error_message'] = str(e)
    
    return jsonify(result)

@app.route('/api/market-data', methods=['GET'])
def get_market_data():
    """
    Main API endpoint with timeframe support
    Query params: timeframe (optional, default: 15m)
    """
    try:
        timeframe = request.args.get('timeframe', '15m')
        
        if timeframe not in TIMEFRAMES:
            return jsonify({'error': 'Invalid timeframe', 'valid_timeframes': list(TIMEFRAMES.keys())}), 400
        
        logger.info(f"Fetching market data for timeframe: {timeframe}")
        
        # Fetch Indian markets
        logger.info("Fetching Nifty raw data...")
        nifty_data_raw = fetch_market_data('^NSEI', timeframe)
        if nifty_data_raw is None:
            logger.error("Nifty raw data fetch returned None")
            return jsonify({'error': 'Unable to fetch Nifty data from Yahoo Finance', 'stage': 'nifty_raw_fetch'}), 502
        
        logger.info("Processing Nifty data (CPR, S/R, indicators)...")
        try:
            nifty_data = process_market_data('^NSEI', timeframe)
        except Exception as e:
            logger.error(f"Nifty processing failed: {e}", exc_info=True)
            return jsonify({'error': f'Nifty processing failed: {str(e)}', 'stage': 'nifty_processing'}), 500
        
        if nifty_data is None:
            return jsonify({'error': 'Nifty data processing returned None', 'stage': 'nifty_processing'}), 502
        
        logger.info("Processing Bank Nifty data...")
        try:
            banknifty_data = process_market_data('^NSEBANK', timeframe)
        except Exception as e:
            logger.error(f"Bank Nifty processing failed: {e}", exc_info=True)
            return jsonify({'error': f'Bank Nifty processing failed: {str(e)}', 'stage': 'banknifty_processing'}), 500
        
        if banknifty_data is None:
            return jsonify({'error': 'Bank Nifty data processing returned None', 'stage': 'banknifty_processing'}), 502
        
        # Fetch global markets (always daily)
        logger.info("Fetching global markets...")
        try:
            global_markets = fetch_global_markets()
        except Exception as e:
            logger.error(f"Global markets fetch failed: {e}", exc_info=True)
            global_markets = {'asian': [], 'european': [], 'us': []}
        
        # Generate prediction
        logger.info("Generating market prediction...")
        try:
            prediction = predict_market_direction(nifty_data_raw, global_markets, nifty_data.get('indicators', {}))
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}", exc_info=True)
            prediction = {'direction': 'neutral', 'confidence': 50.0, 'sentiment': 'neutral', 'signals': {}, 'global_positive_ratio': 50.0}
        
        response = {
            'timeframe': timeframe,
            'timeframe_label': TIMEFRAMES[timeframe]['label'],
            'available_timeframes': {k: v['label'] for k, v in TIMEFRAMES.items()},
            'prediction': prediction,
            'globalMarkets': global_markets,
            'nifty': nifty_data,
            'bankNifty': banknifty_data,
            'timestamp': datetime.now().isoformat(),
            'market_status': get_market_status(),
            'status': 'success'
        }
        
        logger.info(f"Data fetched successfully for {timeframe}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Unhandled error in get_market_data: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/timeframes', methods=['GET'])
def get_timeframes():
    """Get available timeframes"""
    return jsonify({
        'timeframes': TIMEFRAMES,
        'default': '15m'
    })

def get_market_status():
    """Determine if market is open"""
    now = datetime.now()
    
    # Check if weekday (Monday=0 to Friday=4)
    if now.weekday() >= 5:
        return 'closed'
    
    # Indian market hours (9:15 AM to 3:30 PM IST)
    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)
    
    if market_open <= now <= market_close:
        return 'open'
    elif now < market_open:
        return 'pre_market'
    else:
        return 'closed'

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Multi-Timeframe Market Predictor API',
        'market_status': get_market_status()
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'service': 'Indian Stock Market Predictor - Multi-Timeframe',
        'version': '2.0.0',
        'endpoints': {
            '/api/market-data?timeframe=15m': 'GET - Fetch market data',
            '/api/timeframes': 'GET - Get available timeframes',
            '/api/health': 'GET - Health check'
        },
        'timeframes': list(TIMEFRAMES.keys()),
        'features': [
            'Multi-timeframe support (1m to 1wk)',
            'Real-time data from Yahoo Finance',
            'CPR calculation',
            'Support/Resistance levels',
            'Technical indicators (RSI, MACD, Bollinger Bands, etc.)',
            'Volume analysis',
            'Market direction prediction'
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info("Starting Multi-Timeframe Market Predictor API...")
    logger.info("Available timeframes: " + ", ".join(TIMEFRAMES.keys()))
    logger.info(f"Server running on http://0.0.0.0:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)
