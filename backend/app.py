"""
Indian Stock Market Predictor - Multi-Timeframe Backend
Real-time data with multiple timeframe support - Patched for curl_cffi / yfinance cookie crash
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import ta
import logging
import time
import os
from types import SimpleNamespace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
        }
    },
)

# Yahoo Finance configuration
# Do not pass custom curl_cffi sessions or monkey-patched cookie jars to yfinance.
# yfinance manages its own compatible session internally.
CURL_CFFI_AVAILABLE = False
_YF_SESSION = None

def get_yf_session():
    return None

def get_ticker(symbol):
    return yf.Ticker(symbol)

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
        logger.warning(f"CPR period fetch failed for {symbol}: {e}")
        return None

def calculate_cpr_with_period(symbol, timeframe, current_data):
    period_data = get_cpr_period_data(symbol, timeframe)
    if period_data is None:
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
    period_data = get_cpr_period_data(symbol, timeframe)
    if period_data is None:
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
    
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    r4 = high + 3 * (pivot - low)
    
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    s4 = low - 3 * (high - pivot)
    
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
    return support, resistance, {'basis': str(basis), 'period_label': str(period_label)}

def fetch_market_data(symbol, timeframe='15m'):
    """Fetch OHLCV safely without custom sessions or cookie monkey patches."""
    if timeframe not in TIMEFRAMES:
        timeframe = '15m'
    config = TIMEFRAMES[timeframe]

    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                period=config['period'],
                interval=config['interval'],
                timeout=20,
                raise_errors=False
            )
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data
        except Exception as e:
            logger.warning(f"Yahoo history failed for {symbol}, attempt {attempt + 1}: {e}")

        try:
            data = yf.download(
                tickers=symbol, period=config['period'], interval=config['interval'],
                progress=False, auto_adjust=False, threads=False, timeout=20, group_by='column'
            )
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data
        except Exception as e:
            logger.warning(f"Yahoo download failed for {symbol}, attempt {attempt + 1}: {e}")
        time.sleep(1.5 * (attempt + 1))

    logger.error(f"No market data returned for {symbol} ({timeframe})")
    return None

def calculate_volume_analysis(data):
    if data is None or len(data) < 20:
        return {'current_volume': 0, 'avg_volume': 0, 'volume_ratio': 1.0, 'volume_trend': 'neutral'}
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
    except Exception:
        return {'current_volume': 0, 'avg_volume': 0, 'volume_ratio': 1.0, 'volume_trend': 'neutral'}

def generate_candlestick_data(data, max_candles=100):
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
    if data is None or len(data) < 20:
        return {}
    try:
        indicators = {}
        if len(data) >= 14:
            rsi_indicator = ta.momentum.RSIIndicator(data['Close'], window=14)
            rsi_value = rsi_indicator.rsi().iloc[-1]
            if not pd.isna(rsi_value):
                indicators['rsi'] = float(round(rsi_value, 2))
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
        if len(data) >= 12:
            ema_12 = data['Close'].ewm(span=12, adjust=False).mean().iloc[-1]
            if not pd.isna(ema_12):
                indicators['ema_12'] = float(round(ema_12, 2))
        if len(data) >= 26:
            ema_26 = data['Close'].ewm(span=26, adjust=False).mean().iloc[-1]
            if not pd.isna(ema_26):
                indicators['ema_26'] = float(round(ema_26, 2))
        if len(data) >= 26:
            macd_indicator = ta.trend.MACD(data['Close'])
            macd_val = macd_indicator.macd().iloc[-1]
            macd_sig = macd_indicator.macd_signal().iloc[-1]
            macd_diff = macd_indicator.macd_diff().iloc[-1]
            if not pd.isna(macd_val): indicators['macd'] = float(round(macd_val, 2))
            if not pd.isna(macd_sig): indicators['macd_signal'] = float(round(macd_sig, 2))
            if not pd.isna(macd_diff): indicators['macd_diff'] = float(round(macd_diff, 2))
        if len(data) >= 20:
            bb_indicator = ta.volatility.BollingerBands(data['Close'], window=20)
            bb_upper = bb_indicator.bollinger_hband().iloc[-1]
            bb_middle = bb_indicator.bollinger_mavg().iloc[-1]
            bb_lower = bb_indicator.bollinger_lband().iloc[-1]
            if not pd.isna(bb_upper): indicators['bb_upper'] = float(round(bb_upper, 2))
            if not pd.isna(bb_middle): indicators['bb_middle'] = float(round(bb_middle, 2))
            if not pd.isna(bb_lower): indicators['bb_lower'] = float(round(bb_lower, 2))
        if len(data) >= 14:
            atr_indicator = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14)
            atr_val = atr_indicator.average_true_range().iloc[-1]
            if not pd.isna(atr_val):
                indicators['atr'] = float(round(atr_val, 2))
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
        return {'direction': 'neutral', 'confidence': 50.0, 'sentiment': 'neutral', 'signals': {}, 'global_positive_ratio': 50.0}
    
    bullish_signals = 0
    total_signals = 0
    signals = {}
    
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
    
    current_price = float(nifty_data['Close'].iloc[-1])
    if 'sma_20' in indicators:
        if current_price > indicators['sma_20']:
            bullish_signals += 1
            signals['sma'] = 'bullish'
        else:
            signals['sma'] = 'bearish'
        total_signals += 1
    
    if 'macd' in indicators and 'macd_signal' in indicators:
        if indicators['macd'] > indicators['macd_signal']:
            bullish_signals += 1
            signals['macd'] = 'bullish'
        else:
            signals['macd'] = 'bearish'
        total_signals += 1
    
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
    
    positive_global = sum(1 for region in global_markets.values() for market in region if market.get('change', 0) > 0)
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
    
    confidence = (bullish_signals / total_signals) * 100 if total_signals > 0 else 50
    if confidence > 65:
        direction, sentiment = 'bullish', 'positive'
    elif confidence < 35:
        direction, sentiment = 'bearish', 'negative'
    else:
        direction, sentiment = 'neutral', 'neutral'
        
    return {
        'direction': str(direction),
        'confidence': float(round(confidence, 1)),
        'sentiment': str(sentiment),
        'signals': signals,
        'global_positive_ratio': float(round(global_ratio * 100, 1))
    }

def process_market_data(symbol, timeframe='15m', data=None):
    try:
        if data is None:
            data = fetch_market_data(symbol, timeframe)
        if data is None or len(data) == 0:
            return None
        
        current_price = float(data['Close'].iloc[-1])
        prev_close = float(data['Open'].iloc[0])
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
        
        cpr = calculate_cpr_with_period(symbol, timeframe, data)
        support, resistance, sr_info = calculate_support_resistance_with_period(symbol, timeframe, data)
        candles = generate_candlestick_data(data)
        volume = calculate_volume_analysis(data)
        indicators = calculate_technical_indicators(data)
        
        return {
            'current': float(round(current_price, 2)),
            'open': float(round(prev_close, 2)),
            'high': float(round(data['High'].max(), 2)),
            'low': float(round(data['Low'].min(), 2)),
            'change': float(round(change, 2)),
            'changePercent': float(round(change_percent, 2)),
            'cpr': cpr,
            'support': support,
            'resistance': resistance,
            'sr_info': sr_info,
            'cpr_basis': str(TIMEFRAMES.get(timeframe, {}).get('cpr_basis', 'daily')),
            'candleData': candles,
            'volume': volume,
            'indicators': indicators,
            'dataPoints': int(len(data))
        }
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None

def fetch_global_markets():
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
                        'name': str(name), 'value': float(round(current, 2)),
                        'change': float(round(change, 2)), 'trend': 'up' if change > 0 else 'down'
                    })
            except Exception:
                continue
    return global_markets

@app.route('/api/market-data', methods=['GET'])
def get_market_data():
    try:
        timeframe = request.args.get('timeframe', '15m')
        if timeframe not in TIMEFRAMES:
            return jsonify({'error': 'Invalid timeframe'}), 400
        
        nifty_data_raw = fetch_market_data('^NSEI', timeframe)
        if nifty_data_raw is None:
            return jsonify({'error': 'Unable to fetch Nifty data due to provider connection error.'}), 502
        
        nifty_data = process_market_data('^NSEI', timeframe, data=nifty_data_raw)
        banknifty_data = process_market_data('^NSEBANK', timeframe)
        global_markets = fetch_global_markets()
        
        # Safeguard if internal loops generated partial structures
        nifty_indicators = nifty_data.get('indicators', {}) if nifty_data else {}
        prediction = predict_market_direction(nifty_data_raw, global_markets, nifty_indicators)
        
        return jsonify({
            'timeframe': timeframe,
            'timeframe_label': TIMEFRAMES[timeframe]['label'],
            'prediction': prediction,
            'globalMarkets': global_markets,
            'nifty': nifty_data,
            'bankNifty': banknifty_data,
            'timestamp': datetime.now().isoformat(),
            'market_status': get_market_status(),
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Global handler exception: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

def get_market_status():
    now = datetime.now()
    if now.weekday() >= 5: return 'closed'
    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)
    return 'open' if market_open <= now <= market_close else 'closed'

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'service': 'Indian Stock Market Predictor', 'version': '2.0.0-patched'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
