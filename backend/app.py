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
    Tries Ticker.history() first, then falls back to yf.download() which
    uses a different internal code path and sometimes succeeds when the
    Ticker-based approach fails against Yahoo's current API behavior.
