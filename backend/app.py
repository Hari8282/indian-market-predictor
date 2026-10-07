"""
Indian Stock Market Predictor & Quantitative Backtester - Multi-Timeframe Backend
RCBO (Red Candle Breakout) Quantitative Trading Engine and Strategy Backtester.
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import logging
import time
import os
import requests
import json
import threading
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
IST = ZoneInfo("Asia/Kolkata")

def now_ist():
    return datetime.now(IST)

def to_ist_index(index):
    """Make a DatetimeIndex tz-aware in IST, whether it arrived naive or in another tz."""
    if index.tz is None:
        return index.tz_localize(IST)
    return index.tz_convert(IST)

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

# Configuration & Global Constants
_MARKET_CACHE = {}
MARKET_CACHE_TTL = 45

GLOBAL_INDICES = {
    'asian': {'^N225': 'Nikkei 225', '^HSI': 'Hang Seng', '000001.SS': 'Shanghai Composite', '^KS11': 'KOSPI'},
    'european': {'^FTSE': 'FTSE 100', '^GDAXI': 'DAX', '^FCHI': 'CAC 40'},
    'us': {'^GSPC': 'S&P 500', '^IXIC': 'NASDAQ', '^DJI': 'Dow Jones'}
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

SYMBOL_LABELS = {'^NSEI': 'NIFTY 50', '^NSEBANK': 'BANK NIFTY', '^BSESN': 'SENSEX'}

# GitHub-backed 5m History Utilities
def _clean_env(name, default=""):
    val = os.environ.get(name)
    return val.strip().strip('/') if val else default

GITHUB_TOKEN = _clean_env("GITHUB_TOKEN", default=None) or None
GITHUB_REPO = _clean_env("GITHUB_REPO", default=None) or None
GITHUB_BRANCH = _clean_env("GITHUB_BRANCH", default="main")
GITHUB_DATA_DIR = _clean_env("GITHUB_DATA_DIR", default="market_data")
HISTORY_SYMBOLS = {'^NSEI': 'NIFTY', '^NSEBANK': 'BANKNIFTY', '^BSESN': 'SENSEX'}

def github_configured():
    return bool(GITHUB_TOKEN and GITHUB_REPO)

def _history_file_path(symbol):
    name = HISTORY_SYMBOLS.get(symbol, symbol.replace('^', '').replace('/', '_'))
    return f"{GITHUB_DATA_DIR}/{name}_5m.csv"

def github_get_file(path):
    if not github_configured():
        return None, None
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    try:
        r = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH}, timeout=20)
        if r.status_code == 200:
            j = r.json()
            return base64.b64decode(j["content"]).decode("utf-8"), j["sha"]
        return None, None
    except Exception as e:
        logger.warning(f"GitHub read error for {path}: {e}")
        return None, None

def _csv_to_history_df(csv_str):
    df = pd.read_csv(io.StringIO(csv_str))
    if 'Datetime' not in df.columns:
        return None
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True, errors='coerce')
    df = df.dropna(subset=['Datetime']).set_index('Datetime')
    df.index = df.index.tz_convert(IST)
    for col in ('Open', 'High', 'Low', 'Close', 'Volume'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['Open', 'High', 'Low', 'Close']).sort_index()

def load_5m_history(symbol):
    csv_str, _ = github_get_file(_history_file_path(symbol))
    if not csv_str:
        return None
    return _csv_to_history_df(csv_str)

def fetch_5m_history_chunked(symbol, days=60, chunk_days=7):
    days = min(days, 60)
    end = now_ist()
    start_floor = end - timedelta(days=days)
    frames = []
    cursor_end = end
    ticker = yf.Ticker(symbol)

    while cursor_end > start_floor:
        cursor_start = max(start_floor, cursor_end - timedelta(days=chunk_days))
        try:
            chunk = ticker.history(
                start=cursor_start.strftime('%Y-%m-%d'),
                end=(cursor_end + timedelta(days=1)).strftime('%Y-%m-%d'),
                interval='5m', timeout=15
            )
            chunk = _normalize_yf_data(chunk)
            if chunk is not None and len(chunk):
                frames.append(chunk)
        except Exception as e:
            logger.warning(f"5m history fetch failed for {symbol}: {e}")
        cursor_end = cursor_start
        time.sleep(0.2)

    if not frames:
        return None
    combined = pd.concat(frames)
    return combined[~combined.index.duplicated(keep='last')].sort_index()

def _normalize_yf_data(data):
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(data.columns)):
        return None
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    try:
        data.index = to_ist_index(data.index)
    except Exception as e:
        logger.warning(f"Could not normalize index to IST: {e}")
    return data

def _daily_ohlc_from_5m(intraday_df):
    return intraday_df.groupby(intraday_df.index.date).agg(
        Open=('Open', 'first'), High=('High', 'max'), Low=('Low', 'min'), Close=('Close', 'last')
    )

# ---------------------------------------------------------------------------
# Red Candle Breakout (RCBO) Strategy Engine
# ---------------------------------------------------------------------------

RCBO_TRAIL_TIERS = [
    {"trigger": 1.5, "lock": 0.0}, # Lock entry at 1.5R
    {"trigger": 2.0, "lock": 1.0}, # Lock 1.0R at 2.0R
    {"trigger": 3.0, "lock": 2.0}, # Lock 2.0R at 3.0R
    {"trigger": 4.0, "lock": 3.0}, # Lock 3.0R at 4.0R
]

def run_rcbo_backtest(symbol, days=100, min_rr=2.0, max_candle_pct=0.8, filter_cpr=True, trail_tiers=None):
    """
    Backtests the Red Candle Breakout (RCBO) Strategy:
    1. First 5m candle of the day must be bearish (Close < Open).
    2. Entry: Long if a subsequent candle closes above the high of the 1st red candle.
    3. Filter: Reject entry if entry price breaks directly into CPR resistance (between BC & TC).
    4. Stop-Loss: Entry candle Low. If entry candle range > max_candle_pct (%) of price, use PDH.
    5. Trailing Stop & Targets: Tiered trailing stop-loss with initial Target = Min R:R.
    """
    if trail_tiers is None:
        trail_tiers = RCBO_TRAIL_TIERS

    intraday = load_5m_history(symbol)
    data_source = 'github_archive'
    if intraday is None or len(intraday) == 0:
        intraday = fetch_5m_history_chunked(symbol, days=days)
        data_source = 'live_fallback'

    if intraday is None or len(intraday) == 0:
        return {
            "symbol": symbol, "trades": [], "stats": _backtest_stats([]),
            "daysAnalyzed": 0, "setupsIdentified": 0, "dataSource": "unavailable",
            "note": "No 5m history available."
        }

    cutoff = now_ist() - timedelta(days=days)
    intraday = intraday[intraday.index >= cutoff]
    daily_ohlc = _daily_ohlc_from_5m(intraday)
    
    trades = []
    setups_identified = 0
    trading_dates = sorted(set(intraday.index.date))

    for i, d in enumerate(trading_dates):
        if i == 0:
            continue
        prev_date = trading_dates[i - 1]
        if prev_date not in daily_ohlc.index:
            continue

        prev = daily_ohlc.loc[prev_date]
        pdh = float(prev['High'])
        pdl = float(prev['Low'])
        pdc = float(prev['Close'])

        # Calculate CPR Levels
        pivot = (pdh + pdl + pdc) / 3.0
        bc = (pdh + pdl) / 2.0
        tc = (pivot - bc) + pivot
        cpr_bottom = min(bc, tc)
        cpr_top = max(bc, tc)

        day_candles = intraday[intraday.index.date == d]
        if len(day_candles) < 2:
            continue

        # Trigger Condition: First 5-min candle must be Bearish
        first_candle = day_candles.iloc[0]
        first_open = float(first_candle['Open'])
        first_close = float(first_candle['Close'])
        first_high = float(first_candle['High'])

        if first_close >= first_open:
            continue # Skip non-red first candles

        setups_identified += 1
        rest_of_day = day_candles.iloc[1:]

        # Entry Condition: Subsequent candle closes above the high of the 1st red candle
        breakout_candidates = rest_of_day[rest_of_day['Close'] > first_high]
        if len(breakout_candidates) == 0:
            continue

        entry_candle = breakout_candidates.iloc[0]
        entry_time = breakout_candidates.index[0]
        entry_price = float(entry_candle['Close'])
        entry_low = float(entry_candle['Low'])
        entry_high = float(entry_candle['High'])

        # Filter: Ensure breakout does not close directly inside major resistance (CPR)
        if filter_cpr and (cpr_bottom <= entry_price <= cpr_top):
            logger.info(f"[{d}] RCBO Breakout filtered out: Closed inside CPR [{cpr_bottom:.2f} - {cpr_top:.2f}]")
            continue

        # Stop-Loss Selection: Candle Low or PDH if candle is too large
        candle_size_pct = ((entry_high - entry_low) / entry_price) * 100.0
        if candle_size_pct > max_candle_pct and pdh < entry_price:
            stop_loss = pdh
            sl_reason = "PDH (Candle Too Large)"
        else:
            stop_loss = entry_low
            sl_reason = "Entry Candle Low"

        risk = entry_price - stop_loss
        if risk <= 0:
            continue # Invalid risk structure

        target_price = entry_price + (risk * min_rr)

        # Simulate Trailing & Exit Logic for the trade
        trade = _simulate_rcbo_trade(
            day_candles=day_candles,
            entry_time=entry_time,
            entry_price=entry_price,
            initial_stop=stop_loss,
            initial_target=target_price,
            risk=risk,
            trade_date=d,
            sl_reason=sl_reason,
            trail_tiers=trail_tiers
        )
        if trade:
            trades.append(trade)

    return {
        "strategy": "Red Candle Breakout (RCBO)",
        "symbol": symbol,
        "symbolLabel": SYMBOL_LABELS.get(symbol, symbol),
        "trades": trades,
        "stats": _backtest_stats(trades),
        "daysAnalyzed": len(trading_dates) - 1,
        "setupsIdentified": setups_identified,
        "dataSource": data_source,
        "parameters": {
            "minRR": min_rr,
            "maxCandlePct": max_candle_pct,
            "cprFilterApplied": filter_cpr
        }
    }

def _simulate_rcbo_trade(day_candles, entry_time, entry_price, initial_stop, initial_target, risk, trade_date, sl_reason, trail_tiers):
    after_entry = day_candles[day_candles.index > entry_time]
    exit_price, exit_reason, exit_time = None, 'EOD_CLOSE', None
    current_stop, current_target = initial_stop, initial_target
    trailed = False

    for t, row in after_entry.iterrows():
        high, low = float(row['High']), float(row['Low'])

        # Check Stop Loss Hit
        if low <= current_stop:
            exit_price = current_stop
            exit_reason = 'TRAILED_STOP' if trailed else 'STOPPED_OUT'
            exit_time = t
            break

        # Check Target Hit
        if high >= current_target:
            exit_price = current_target
            exit_reason = 'TARGET_HIT'
            exit_time = t
            break

        # Trailing Logic
        current_r = (high - entry_price) / risk
        for tier in sorted(trail_tiers, key=lambda x: x["trigger"], reverse=True):
            if current_r >= tier["trigger"]:
                new_stop = entry_price + (tier["lock"] * risk)
                if new_stop > current_stop:
                    current_stop = new_stop
                    current_target = entry_price + (tier["trigger"] + 2.0) * risk
                    trailed = True
                break

    if exit_price is None:
        exit_price = float(day_candles.iloc[-1]['Close'])
        exit_time = day_candles.index[-1]

    pnl = exit_price - entry_price
    achieved_rr = round(pnl / risk, 2)

    return {
        "date": trade_date.strftime('%Y-%m-%d'),
        "signal": "BUY",
        "entryTime": entry_time.strftime('%H:%M'),
        "exitTime": exit_time.strftime('%H:%M') if exit_time else None,
        "entry": round(entry_price, 2),
        "stopLoss": round(current_stop, 2),
        "target": round(current_target, 2),
        "exitPrice": round(exit_price, 2),
        "riskReward": achieved_rr,
        "status": exit_reason,
        "pnlPoints": round(pnl, 2),
        "slType": sl_reason,
        "trailed": trailed
    }

def _backtest_stats(trades):
    if not trades:
        return {
            "totalTrades": 0, "wins": 0, "losses": 0, "winRate": 0.0,
            "totalPnlPoints": 0.0, "avgPnlPoints": 0.0, "avgRiskReward": 0.0,
            "targetHits": 0, "stopOuts": 0, "trailedStops": 0, "eodCloses": 0
        }
    wins = [t for t in trades if t['pnlPoints'] > 0]
    losses = [t for t in trades if t['pnlPoints'] <= 0]
    total_pnl = sum(t['pnlPoints'] for t in trades)
    rr_values = [t['riskReward'] for t in trades]
    return {
        "totalTrades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "winRate": round((len(wins) / len(trades)) * 100, 1),
        "totalPnlPoints": round(total_pnl, 2),
        "avgPnlPoints": round(total_pnl / len(trades), 2),
        "avgRiskReward": round(sum(rr_values) / len(rr_values), 2) if rr_values else 0.0,
        "targetHits": sum(1 for t in trades if t['status'] == 'TARGET_HIT'),
        "stopOuts": sum(1 for t in trades if t['status'] == 'STOPPED_OUT'),
        "trailedStops": sum(1 for t in trades if t['status'] == 'TRAILED_STOP'),
        "eodCloses": sum(1 for t in trades if t['status'] == 'EOD_CLOSE')
    }

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route('/api/backtest/rcbo', methods=['GET', 'POST'])
def api_backtest_rcbo():
    """API Endpoint to trigger RCBO Backtest."""
    if request.method == 'POST':
        params = request.get_json() or {}
    else:
        params = request.args

    symbol = params.get('symbol', '^NSEI')
    days = int(params.get('days', 60))
    min_rr = float(params.get('min_rr', 2.0))
    max_candle_pct = float(params.get('max_candle_pct', 0.8))
    filter_cpr = str(params.get('filter_cpr', 'true')).lower() in ['true', '1', 'yes']

    results = run_rcbo_backtest(
        symbol=symbol,
        days=days,
        min_rr=min_rr,
        max_candle_pct=max_candle_pct,
        filter_cpr=filter_cpr
    )
    return jsonify(results)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "timestamp": now_ist().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
