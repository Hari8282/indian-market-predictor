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
import requests
import json
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

_MARKET_CACHE = {}
MARKET_CACHE_TTL = 45

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

def _normalize_yf_data(data):
    """Normalize Yahoo/yfinance data to a simple OHLCV DataFrame."""
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(data.columns)):
        return None
    return data.dropna(subset=["Open", "High", "Low", "Close"])


def _fetch_yahoo_chart_direct(symbol, period, interval):
    """Fallback that calls Yahoo's chart API directly when yfinance fails."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": period, "interval": interval, "includePrePost": "false"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        payload = response.json()
        result = payload.get("chart", {}).get("result")
        if not result:
            return None
        result = result[0]
        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        if not timestamps:
            return None
        df = pd.DataFrame({
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", [0] * len(timestamps)),
        }, index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None))
        return _normalize_yf_data(df)
    except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"Direct Yahoo chart fallback failed for {symbol}: {e}")
        return None


def fetch_market_data(symbol, timeframe='15m'):
    """Fetch market data with retries, direct Yahoo fallback, and short cache."""
    if timeframe not in TIMEFRAMES:
        timeframe = '15m'

    cache_key = (symbol, timeframe)
    cached = _MARKET_CACHE.get(cache_key)
    if cached and (time.time() - cached["timestamp"] < MARKET_CACHE_TTL):
        return cached["data"].copy()

    config = TIMEFRAMES[timeframe]

    for attempt in range(2):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                period=config['period'],
                interval=config['interval'],
                timeout=15,
                raise_errors=False,
            )
            data = _normalize_yf_data(data)
            if data is not None and not data.empty:
                _MARKET_CACHE[cache_key] = {"timestamp": time.time(), "data": data.copy()}
                return data
        except Exception as e:
            logger.warning(f"yfinance history failed for {symbol}: {e}")

        try:
            data = yf.download(
                tickers=symbol,
                period=config['period'],
                interval=config['interval'],
                progress=False,
                threads=False,
                timeout=15,
                auto_adjust=False,
            )
            data = _normalize_yf_data(data)
            if data is not None and not data.empty:
                _MARKET_CACHE[cache_key] = {"timestamp": time.time(), "data": data.copy()}
                return data
        except Exception as e:
            logger.warning(f"yfinance download failed for {symbol}: {e}")

        data = _fetch_yahoo_chart_direct(
            symbol,
            config['period'],
            config['interval'],
        )
        if data is not None and not data.empty:
            logger.info(f"Direct Yahoo fallback succeeded for {symbol}")
            _MARKET_CACHE[cache_key] = {"timestamp": time.time(), "data": data.copy()}
            return data

        if attempt == 0:
            time.sleep(1)

    logger.error(f"All market-data providers failed for {symbol}")
    return None


MIN_STRUCTURE_CANDLES = 20

def detect_market_structure(data, lookback=100, swing=2, min_candles=MIN_STRUCTURE_CANDLES):
    """
    Confirm HH/HL/LH/LL using at least 20 completed candles.
    A pivot is confirmed only after `swing` candles have formed to its right.
    """
    empty = {
        "valid": False, "minimumCandles": min_candles, "candlesAnalyzed": 0,
        "trend": "neutral", "structure": [], "swingHighs": [], "swingLows": [],
        "lastHighType": None, "lastLowType": None, "score": 0
    }
    if data is None or len(data) < min_candles:
        empty["candlesAnalyzed"] = 0 if data is None else len(data)
        return empty

    d = data.tail(max(min_candles, lookback)).copy()
    highs, lows = [], []

    for i in range(swing, len(d) - swing):
        h = float(d["High"].iloc[i])
        l = float(d["Low"].iloc[i])
        hwin = d["High"].iloc[i-swing:i+swing+1]
        lwin = d["Low"].iloc[i-swing:i+swing+1]

        if h == float(hwin.max()) and h > float(d["High"].iloc[i-1]):
            highs.append((i, h))
        if l == float(lwin.min()) and l < float(d["Low"].iloc[i-1]):
            lows.append((i, l))

    swing_highs, swing_lows = [], []
    for n, (i, price) in enumerate(highs):
        if n == 0:
            typ = "SH"
        else:
            typ = "HH" if price > highs[n-1][1] else "LH"
        swing_highs.append({
            "index": int(i),
            "timestamp": d.index[i].strftime("%Y-%m-%d %H:%M"),
            "price": round(price, 2),
            "type": typ,
            "confirmed": True
        })

    for n, (i, price) in enumerate(lows):
        if n == 0:
            typ = "SL"
        else:
            typ = "HL" if price > lows[n-1][1] else "LL"
        swing_lows.append({
            "index": int(i),
            "timestamp": d.index[i].strftime("%Y-%m-%d %H:%M"),
            "price": round(price, 2),
            "type": typ,
            "confirmed": True
        })

    last_high = swing_highs[-1]["type"] if swing_highs else None
    last_low = swing_lows[-1]["type"] if swing_lows else None

    if last_high == "HH" and last_low == "HL":
        trend, score = "bullish", 2
    elif last_high == "LH" and last_low == "LL":
        trend, score = "bearish", -2
    else:
        trend, score = "neutral", 0

    structure = sorted(
        [x for x in swing_highs + swing_lows if x["type"] in ("HH", "HL", "LH", "LL")],
        key=lambda x: x["index"]
    )[-20:]

    return {
        "valid": True,
        "minimumCandles": min_candles,
        "candlesAnalyzed": len(d),
        "trend": trend,
        "structure": structure,
        "swingHighs": swing_highs[-10:],
        "swingLows": swing_lows[-10:],
        "lastHighType": last_high,
        "lastLowType": last_low,
        "score": score
    }


def _unique_zone_levels(points, current_price, side, tolerance):
    values = sorted([float(x["price"]) for x in points])
    if side == "support":
        values = [v for v in values if v < current_price]
        values.sort(reverse=True)
    else:
        values = [v for v in values if v > current_price]

    merged = []
    for value in values:
        if not any(abs(value - existing) <= tolerance for existing in merged):
            merged.append(value)
    return merged[:4]


def calculate_structure_sr(data, structure):
    if data is None or len(data) == 0 or not structure.get("valid"):
        return [], []

    price = float(data["Close"].iloc[-1])
    recent_range = float((data["High"].tail(20) - data["Low"].tail(20)).mean())
    tolerance = max(recent_range * 0.35, price * 0.001)

    supports = _unique_zone_levels(
        structure.get("swingLows", []), price, "support", tolerance
    )
    resistances = _unique_zone_levels(
        structure.get("swingHighs", []), price, "resistance", tolerance
    )

    return (
        [{"level": f"MS-S{i+1}", "value": round(v, 2), "type": "Market Structure",
          "source": "HL/LL confirmed swing low"} for i, v in enumerate(supports)],
        [{"level": f"MS-R{i+1}", "value": round(v, 2), "type": "Market Structure",
          "source": "HH/LH confirmed swing high"} for i, v in enumerate(resistances)]
    )


def generate_trade_signal(data, prediction, cpr, support, resistance, structure):
    """
    Strict confluence strategy:
      minimum 20 candles + confirmed HH/HL/LH/LL + prediction + CPR +
      market-structure support/resistance + candle confirmation.
    """
    if data is None or len(data) < MIN_STRUCTURE_CANDLES:
        return {
            "signal": "HOLD", "strength": "weak", "confidence": 0, "score": 0,
            "entry": None, "stopLoss": None, "target": None,
            "marketStructure": "neutral",
            "reasons": [f"Waiting for at least {MIN_STRUCTURE_CANDLES} candles"],
            "conditions": {}
        }

    prediction = prediction or {"direction": "neutral", "confidence": 0}
    price = float(data["Close"].iloc[-1])
    trend = structure.get("trend", "neutral")
    direction = prediction.get("direction", "neutral")
    bc, tc = float(cpr.get("bc", 0)), float(cpr.get("tc", 0))
    cpr_low, cpr_high = min(bc, tc), max(bc, tc)

    bullish_candle = float(data["Close"].iloc[-1]) > float(data["Open"].iloc[-1])
    bearish_candle = float(data["Close"].iloc[-1]) < float(data["Open"].iloc[-1])

    structure_ok_buy = trend == "bullish"
    structure_ok_sell = trend == "bearish"
    prediction_buy = direction == "bullish"
    prediction_sell = direction == "bearish"
    cpr_buy = cpr_high > 0 and price > cpr_high
    cpr_sell = cpr_high > 0 and price < cpr_low

    structure_support = [x for x in support if x.get("level", "").startswith("MS-S")]
    structure_resistance = [x for x in resistance if x.get("level", "").startswith("MS-R")]
    nearest_support = max(structure_support, key=lambda x: x["value"], default=None)
    nearest_resistance = min(structure_resistance, key=lambda x: x["value"], default=None)

    buy_score = sum([prediction_buy * 2, structure_ok_buy * 3, cpr_buy * 2, bullish_candle * 1])
    sell_score = sum([prediction_sell * 2, structure_ok_sell * 3, cpr_sell * 2, bearish_candle * 1])

    # Require structure + prediction + CPR alignment. Candle adds confirmation.
    if structure_ok_buy and prediction_buy and cpr_buy and buy_score >= 7:
        signal, score = "BUY", buy_score
    elif structure_ok_sell and prediction_sell and cpr_sell and sell_score >= 7:
        signal, score = "SELL", -sell_score
    else:
        signal, score = "HOLD", 0

    recent_range = float((data["High"].tail(14) - data["Low"].tail(14)).mean())
    risk_buffer = max(recent_range * 0.6, price * 0.002)

    if signal == "BUY":
        stop = (nearest_support["value"] - risk_buffer) if nearest_support else price - risk_buffer
        target = nearest_resistance["value"] if nearest_resistance else price + risk_buffer * 2
    elif signal == "SELL":
        stop = (nearest_resistance["value"] + risk_buffer) if nearest_resistance else price + risk_buffer
        target = nearest_support["value"] if nearest_support else price - risk_buffer * 2
    else:
        stop = target = None

    reasons = []
    if prediction_buy or prediction_sell: reasons.append(f"Prediction: {direction}")
    if trend != "neutral": reasons.append(f"Structure confirmed: {trend} ({structure.get('lastHighType')} + {structure.get('lastLowType')})")
    if cpr_buy: reasons.append("Price above CPR")
    elif cpr_sell: reasons.append("Price below CPR")
    else: reasons.append("Price inside/near CPR")
    if bullish_candle: reasons.append("Latest candle bullish")
    elif bearish_candle: reasons.append("Latest candle bearish")

    confidence = min(95, 50 + abs(score) * 5) if signal != "HOLD" else 35

    return {
        "signal": signal,
        "strength": "strong" if abs(score) >= 8 else "moderate" if abs(score) >= 7 else "weak",
        "confidence": round(confidence, 1),
        "score": score,
        "entry": round(price, 2),
        "stopLoss": round(stop, 2) if stop is not None else None,
        "target": round(target, 2) if target is not None else None,
        "nearestSupport": nearest_support,
        "nearestResistance": nearest_resistance,
        "marketStructure": trend,
        "reasons": reasons,
        "conditions": {
            "minimum20Candles": structure.get("valid", False),
            "prediction": direction,
            "structure": trend,
            "cprPosition": "above" if cpr_buy else "below" if cpr_sell else "inside",
            "candle": "bullish" if bullish_candle else "bearish" if bearish_candle else "doji"
        }
    }

def generate_candlestick_data(data, max_candles=150):
    """Chart-ready OHLCV data. Frontend can render candles, volume and overlays."""
    if data is None or len(data) == 0:
        return []
    d = data.tail(max_candles)
    candles = []
    for idx, row in d.iterrows():
        candles.append({
            "time": int(idx.timestamp()) if hasattr(idx, "timestamp") else str(idx),
            "timestamp": idx.strftime("%Y-%m-%d %H:%M"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(float(row.get("Volume", 0)))
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


def calculate_volume_analysis(data, lookback=20):
    """Safe volume analysis for Yahoo Finance OHLCV data."""
    default = {
        "current_volume": 0,
        "avg_volume": 0,
        "volume_ratio": 1.0,
        "volume_trend": "neutral",
        "volume_signal": "normal"
    }
    try:
        if data is None or data.empty or "Volume" not in data.columns:
            return default

        volume = pd.to_numeric(data["Volume"], errors="coerce").fillna(0.0)
        current_volume = float(volume.iloc[-1])
        history = volume.iloc[max(0, len(volume)-lookback-1):-1]
        if history.empty:
            history = volume.iloc[:-1]
        avg_volume = float(history.mean()) if not history.empty else current_volume

        if avg_volume <= 0:
            ratio = 1.0
        else:
            ratio = current_volume / avg_volume

        recent = volume.tail(min(5, len(volume))).mean()
        previous = volume.iloc[max(0, len(volume)-10):max(0, len(volume)-5)].mean()
        if previous and previous > 0:
            trend = "increasing" if recent > previous * 1.05 else "decreasing" if recent < previous * 0.95 else "neutral"
        else:
            trend = "neutral"

        signal = "high" if ratio >= 1.5 else "low" if ratio <= 0.6 else "normal"
        return {
            "current_volume": int(round(current_volume)),
            "avg_volume": int(round(avg_volume)),
            "volume_ratio": round(float(ratio), 2),
            "volume_trend": trend,
            "volume_signal": signal
        }
    except Exception as e:
        logger.warning("Volume analysis failed: %s", e)
        return default

def process_market_data(symbol, timeframe='15m', prediction=None, data=None):
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
        market_structure = detect_market_structure(data)
        structure_support, structure_resistance = calculate_structure_sr(data, market_structure)
        support = structure_support + support
        resistance = structure_resistance + resistance
        candles = generate_candlestick_data(data)
        volume = calculate_volume_analysis(data)
        indicators = calculate_technical_indicators(data)
        if prediction is None:
            prediction = {'direction': 'neutral', 'confidence': 50.0}
        trade_signal = generate_trade_signal(data, prediction, cpr, support, resistance, market_structure)
        chart_overlays = {
            'cpr': [
                {'name': 'BC', 'value': cpr.get('bc')},
                {'name': 'PIVOT', 'value': cpr.get('pivot')},
                {'name': 'TC', 'value': cpr.get('tc')}
            ],
            'support': support,
            'resistance': resistance,
            'structureMarkers': market_structure.get('structure', [])
        }
        
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
            'marketStructure': market_structure,
            'tradeSignal': trade_signal,
            'chartOverlays': chart_overlays,
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

        # Fetch Nifty once and reuse the same dataframe throughout this request.
        nifty_data_raw = fetch_market_data('^NSEI', timeframe)

        # Global markets are used to calculate the market prediction.
        global_markets = fetch_global_markets()

        nifty_indicators = (
            calculate_technical_indicators(nifty_data_raw)
            if nifty_data_raw is not None else {}
        )

        prediction = predict_market_direction(
            nifty_data_raw,
            global_markets,
            nifty_indicators
        )

        nifty_data = process_market_data(
            '^NSEI',
            timeframe,
            prediction=prediction,
            data=nifty_data_raw
        ) if nifty_data_raw is not None else None

        banknifty_data = process_market_data(
            '^NSEBANK',
            timeframe,
            prediction=prediction
        )

        provider_available = nifty_data_raw is not None

        return jsonify({
            'timeframe': timeframe,
            'timeframe_label': TIMEFRAMES[timeframe]['label'],
            'prediction': prediction,
            'globalMarkets': global_markets,
            'nifty': nifty_data,
            'bankNifty': banknifty_data,
            'timestamp': datetime.now().isoformat(),
            'market_status': get_market_status(),
            'status': 'success' if provider_available else 'degraded',
            'data_provider_status': (
                'available' if provider_available else 'temporarily_unavailable'
            )
        })
    except Exception as e:
        logger.exception("Global handler exception")
        return jsonify({
            'error': 'Internal market data processing error',
            'details': str(e),
            'status': 'error'
        }), 500

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
