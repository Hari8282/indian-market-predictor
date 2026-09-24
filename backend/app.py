"""
Indian Stock Market Predictor - Multi-Timeframe Backend
Real-time data with multiple timeframe support - Patched for curl_cffi / yfinance cookie crash
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import ta
import logging
import time
import os
import requests
import json
import threading
import io
import base64
from types import SimpleNamespace

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

# Yahoo Finance configuration
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

# ---------------------------------------------------------------------------
# Buy/Sell Signal Log (Trading Journal)
# ---------------------------------------------------------------------------
SIGNAL_LOG_LOCK = threading.Lock()
SIGNAL_LOG = {'^NSEI': [], '^NSEBANK': []}
_SIGNAL_LOG_COUNTER = 0
MAX_SIGNAL_LOG_PER_SYMBOL = 200
SYMBOL_LABELS = {'^NSEI': 'NIFTY 50', '^NSEBANK': 'BANK NIFTY', '^BSESN': 'SENSEX'}

# ---------------------------------------------------------------------------
# GitHub-backed 5m history archive
# ---------------------------------------------------------------------------
GITHUB_API = "https://api.github.com"

def _clean_env(name, default=""):
    val = os.environ.get(name)
    if val is None:
        return default
    val = val.strip().strip('/')
    return val if val else default

GITHUB_TOKEN = _clean_env("GITHUB_TOKEN", default=None) or None
GITHUB_REPO = _clean_env("GITHUB_REPO", default=None) or None
GITHUB_BRANCH = _clean_env("GITHUB_BRANCH", default="main")
GITHUB_DATA_DIR = _clean_env("GITHUB_DATA_DIR", default="market_data")
HISTORY_RETENTION_DAYS = 100
HISTORY_SYMBOLS = {'^NSEI': 'NIFTY', '^NSEBANK': 'BANKNIFTY', '^BSESN': 'SENSEX'}

def github_configured():
    return bool(GITHUB_TOKEN and GITHUB_REPO)

def _github_headers():
    return {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

def _history_file_path(symbol):
    name = HISTORY_SYMBOLS.get(symbol, symbol.replace('^', '').replace('/', '_'))
    segments = [s.strip().strip('/') for s in (GITHUB_DATA_DIR, f"{name}_5m.csv")]
    return "/".join(s for s in segments if s)

def github_get_file(path):
    if not github_configured():
        return None, None
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{path}"
    try:
        r = requests.get(url, headers=_github_headers(), params={"ref": GITHUB_BRANCH}, timeout=20)
        if r.status_code == 200:
            j = r.json()
            content = base64.b64decode(j["content"]).decode("utf-8")
            return content, j["sha"]
        if r.status_code == 404:
            return None, None
        logger.warning(f"GitHub read failed for {path}: {r.status_code} {r.text[:200]}")
        return None, None
    except requests.RequestException as e:
        logger.warning(f"GitHub read error for {path}: {e}")
        return None, None

def github_put_file(path, content_str, message, sha=None):
    if not github_configured():
        raise RuntimeError("GITHUB_TOKEN and/or GITHUB_REPO are not configured on the server")
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{path}"
    payload = {
        "message": message,
        "content": base64.b64encode(content_str.encode("utf-8")).decode("utf-8"),
        "branch": GITHUB_BRANCH
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=_github_headers(), json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub write failed ({r.status_code}): {r.text[:300]}")
    return r.json()

def _history_df_to_csv(df):
    out = io.StringIO()
    df.reset_index().rename(columns={df.index.name or 'index': 'Datetime'}).to_csv(out, index=False)
    return out.getvalue()

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

def fetch_5m_history_chunked(symbol, days=60, chunk_days=7):
    days = min(days, 60)
    end = now_ist()
    start_floor = end - timedelta(days=days)
    frames = []
    cursor_end = end
    ticker = get_ticker(symbol)

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
            logger.warning(f"5m history chunk failed for {symbol} [{cursor_start.date()} .. {cursor_end.date()}]: {e}")
        cursor_end = cursor_start
        time.sleep(0.25)

    if not frames:
        return None
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    return combined

def sync_5m_history_to_github(symbol):
    path = _history_file_path(symbol)
    existing_csv, sha = github_get_file(path)
    existing_df = _csv_to_history_df(existing_csv) if existing_csv else None

    fresh_df = fetch_5m_history_chunked(symbol, days=60)

    if fresh_df is None and existing_df is None:
        return {"symbol": symbol, "status": "no_data", "rows": 0}

    parts = [d for d in (existing_df, fresh_df) if d is not None and len(d)]
    if not parts:
        return {"symbol": symbol, "status": "no_data", "rows": 0}

    merged = pd.concat(parts)
    merged = merged[~merged.index.duplicated(keep='last')].sort_index()

    cutoff = now_ist() - timedelta(days=HISTORY_RETENTION_DAYS)
    merged = merged[merged.index >= cutoff]

    if len(merged) == 0:
        return {"symbol": symbol, "status": "no_data", "rows": 0}

    csv_str = _history_df_to_csv(merged)
    try:
        github_put_file(path, csv_str, f"Update {HISTORY_SYMBOLS.get(symbol, symbol)} 5m history ({len(merged)} rows)", sha=sha)
    except RuntimeError as e:
        return {"symbol": symbol, "status": "error", "error": str(e)}

    trading_days = pd.Series(merged.index.date).nunique()
    return {
        "symbol": symbol,
        "status": "ok",
        "rows": int(len(merged)),
        "tradingDays": int(trading_days),
        "from": merged.index[0].strftime('%Y-%m-%d'),
        "to": merged.index[-1].strftime('%Y-%m-%d')
    }

def load_5m_history(symbol):
    csv_str, _ = github_get_file(_history_file_path(symbol))
    if not csv_str:
        return None
    return _csv_to_history_df(csv_str)

def fetch_global_daily_history(period='400d'):
    out = {}
    for region, indices in GLOBAL_INDICES.items():
        for sym in indices:
            try:
                df = _normalize_yf_data(get_ticker(sym).history(period=period, interval='1d', timeout=15))
                if df is not None and len(df):
                    out[sym] = df
            except Exception as e:
                logger.warning(f"Global daily history failed for {sym}: {e}")
    return out

def historical_global_status(global_daily, target_date):
    positive, total = 0, 0
    for sym, df in global_daily.items():
        try:
            asof = df[df.index.date <= target_date]
            if len(asof) < 2:
                continue
            current = float(asof['Close'].iloc[-1])
            prev = float(asof['Close'].iloc[-2])
            total += 1
            if prev > 0 and current > prev:
                positive += 1
        except Exception:
            continue
    if total == 0:
        return 'neutral'
    ratio = positive / total
    if ratio > 0.6:
        return 'bullish'
    if ratio < 0.4:
        return 'bearish'
    return 'neutral'

def _close_trade(trade, exit_price, status, exit_time):
    trade['status'] = status
    trade['exitPrice'] = round(float(exit_price), 2)
    try:
        trade['closedAt'] = exit_time.strftime('%Y-%m-%d %H:%M:%S')
    except AttributeError:
        trade['closedAt'] = now_ist().isoformat()

    if trade['signal'] == 'BUY':
        pnl = exit_price - trade['entry']
    else:
        pnl = trade['entry'] - exit_price

    trade['pnlPoints'] = round(float(pnl), 2)
    trade['pnlPercent'] = round((pnl / trade['entry']) * 100, 2) if trade.get('entry') else 0.0

def record_signal_for_journal(symbol, timeframe, trade_signal, data):
    global _SIGNAL_LOG_COUNTER
    if data is None or len(data) == 0 or not isinstance(trade_signal, dict):
        return

    try:
        last_high = float(data['High'].iloc[-1])
        last_low = float(data['Low'].iloc[-1])
        last_close = float(data['Close'].iloc[-1])
        last_time = data.index[-1]
    except (KeyError, IndexError, ValueError, TypeError):
        return

    new_signal = trade_signal.get('signal')

    with SIGNAL_LOG_LOCK:
        entries = SIGNAL_LOG.setdefault(symbol, [])
        open_trade = next((t for t in reversed(entries) if t['status'] == 'OPEN'), None)

        if open_trade:
            if open_trade['signal'] == 'BUY':
                if open_trade['stopLoss'] is not None and last_low <= open_trade['stopLoss']:
                    _close_trade(open_trade, open_trade['stopLoss'], 'STOPPED_OUT', last_time)
                    open_trade = None
                elif open_trade['target'] is not None and last_high >= open_trade['target']:
                    _close_trade(open_trade, open_trade['target'], 'TARGET_HIT', last_time)
                    open_trade = None
            elif open_trade['signal'] == 'SELL':
                if open_trade['stopLoss'] is not None and last_high >= open_trade['stopLoss']:
                    _close_trade(open_trade, open_trade['stopLoss'], 'STOPPED_OUT', last_time)
                    open_trade = None
                elif open_trade['target'] is not None and last_low <= open_trade['target']:
                    _close_trade(open_trade, open_trade['target'], 'TARGET_HIT', last_time)
                    open_trade = None

        if open_trade and new_signal != open_trade['signal']:
            _close_trade(open_trade, last_close, 'CLOSED_SIGNAL_CHANGE', last_time)
            open_trade = None

        if not open_trade and new_signal in ('BUY', 'SELL'):
            _SIGNAL_LOG_COUNTER += 1
            entries.append({
                'id': _SIGNAL_LOG_COUNTER,
                'symbol': symbol,
                'symbolLabel': SYMBOL_LABELS.get(symbol, symbol),
                'timeframe': timeframe,
                'signal': new_signal,
                'entry': trade_signal.get('entry'),
                'stopLoss': trade_signal.get('stopLoss'),
                'target': trade_signal.get('target'),
                'riskReward': trade_signal.get('riskReward'),
                'confidence': trade_signal.get('confidence'),
                'reason': trade_signal.get('reason'),
                'openedAt': now_ist().isoformat(),
                'closedAt': None,
                'status': 'OPEN',
                'exitPrice': None,
                'pnlPoints': None,
                'pnlPercent': None
            })

        if len(entries) > MAX_SIGNAL_LOG_PER_SYMBOL:
            SIGNAL_LOG[symbol] = entries[-MAX_SIGNAL_LOG_PER_SYMBOL:]

def _daily_ohlc_from_5m(intraday_df):
    daily = intraday_df.groupby(intraday_df.index.date).agg(
        Open=('Open', 'first'), High=('High', 'max'), Low=('Low', 'min'), Close=('Close', 'last')
    )
    return daily

ENTRY_RANGE_MIN = 0.40
ENTRY_RANGE_MAX = 0.60
STOP_LOSS_BUFFER_PCT = 0.1  # percent; buffer = 0.1% of the entry candle's High/Low (per signal direction)

# ---------------------------------------------------------------------------
# UPDATED: Multi-Tier Trailing Stop Definitions
# ---------------------------------------------------------------------------
TRAIL_TIERS = [
    {"trigger": 1.5, "lock": 0.0},  # > 1.5R profit -> Stop = Entry (0R)
    {"trigger": 2.0, "lock": 0.5},  # > 2.0R profit -> Stop = +0.5R
    {"trigger": 3.0, "lock": 1.5},  # > 3.0R profit -> Stop = +1.5R
    {"trigger": 4.0, "lock": 3.0},  # > 4.0R profit -> Stop = +3.0R
]
EXTENDED_TARGET_R = 10.0

def _choose_stop_loss(signal, entry, entry_row, bc, tc, buffer_pct=STOP_LOSS_BUFFER_PCT):
    """
    Stop loss buffer is 0.1% (configurable) of the entry candle's High/Low,
    applied on the side of the candle that matches the signal direction:
    - BUY: buffer below the entry candle's Low
    - SELL: buffer above the entry candle's High
    """
    if signal == 'BUY':
        low = float(entry_row['Low'])
        buffer_points = low * (buffer_pct / 100.0)
        candidates = [low - buffer_points, bc]
        valid = [c for c in candidates if c < entry]
        return max(valid) if valid else None
    else:
        high = float(entry_row['High'])
        buffer_points = high * (buffer_pct / 100.0)
        candidates = [high + buffer_points, tc]
        valid = [c for c in candidates if c > entry]
        return min(valid) if valid else None

def run_backtest(symbol, days=100, min_rr=2.0, entry_fraction=None, stop_buffer=STOP_LOSS_BUFFER_PCT,
                 trail_tiers=None, extended_target_r=EXTENDED_TARGET_R):
    if trail_tiers is None:
        trail_tiers = TRAIL_TIERS

    intraday = load_5m_history(symbol)
    data_source = 'github_archive'
    if intraday is None or len(intraday) == 0:
        intraday = fetch_5m_history_chunked(symbol, days=60)
        data_source = 'live_fallback_max_60d'
    if intraday is None or len(intraday) == 0:
        return {
            "symbol": symbol, "trades": [], "stats": _backtest_stats([]),
            "daysAnalyzed": 0, "setupsIdentified": 0, "dataSource": "unavailable",
            "note": "No 5m history available yet. Run /api/history/sync first (requires GITHUB_TOKEN/GITHUB_REPO), or wait for the live fallback to have data."
        }

    cutoff = now_ist() - timedelta(days=days)
    intraday = intraday[intraday.index >= cutoff]
    if len(intraday) == 0:
        return {
            "symbol": symbol, "trades": [], "stats": _backtest_stats([]),
            "daysAnalyzed": 0, "setupsIdentified": 0, "dataSource": data_source,
            "note": "No candles fall within the requested day range."
        }

    daily_ohlc = _daily_ohlc_from_5m(intraday)
    global_daily = fetch_global_daily_history()

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
        pivot = (float(prev['High']) + float(prev['Low']) + float(prev['Close'])) / 3
        bc = (float(prev['High']) + float(prev['Low'])) / 2
        tc = (pivot - bc) + pivot

        day_candles = intraday[intraday.index.date == d]
        if len(day_candles) < 2:
            continue

        first = day_candles.iloc[0]
        rest = day_candles.iloc[1:]
        global_status = historical_global_status(global_daily, d)
        first_close = float(first['Close'])
        first_high = float(first['High'])
        first_low = float(first['Low'])
        first_range = first_high - first_low

        first_green = first_close > float(first['Open'])
        first_red = first_close < float(first['Open'])
        above_tc = first_close > tc
        below_bc = first_close < bc
        first_candle_label = 'green' if first_green else ('red' if first_red else 'flat')

        trade = None
        if global_status == 'bullish' and first_green and above_tc:
            setups_identified += 1
            entry_zone_low = first_low + ENTRY_RANGE_MIN * first_range
            entry_zone_high = first_low + ENTRY_RANGE_MAX * first_range
            trigger = rest[(rest['Close'] >= entry_zone_low) & (rest['Close'] <= entry_zone_high)]
            if len(trigger):
                entry_row = trigger.iloc[0]
                entry_time = trigger.index[0]
                entry = float(entry_row['Close'])
                stop = _choose_stop_loss('BUY', entry, entry_row, bc, tc, stop_buffer)
                if stop is not None:
                    risk = entry - stop
                    if risk > 0:
                        target = round(entry + risk * min_rr, 2)
                        trade = _simulate_trade(day_candles, entry_time, 'BUY', entry, stop, target, risk, d, global_status, first_candle_label,
                                                 trail_tiers, extended_target_r)

        elif global_status == 'bearish' and first_red and below_bc:
            setups_identified += 1
            entry_zone_low = first_high - ENTRY_RANGE_MAX * first_range
            entry_zone_high = first_high - ENTRY_RANGE_MIN * first_range
            trigger = rest[(rest['Close'] >= entry_zone_low) & (rest['Close'] <= entry_zone_high)]
            if len(trigger):
                entry_row = trigger.iloc[0]
                entry_time = trigger.index[0]
                entry = float(entry_row['Close'])
                stop = _choose_stop_loss('SELL', entry, entry_row, bc, tc, stop_buffer)
                if stop is not None:
                    risk = stop - entry
                    if risk > 0:
                        target = round(entry - risk * min_rr, 2)
                        trade = _simulate_trade(day_candles, entry_time, 'SELL', entry, stop, target, risk, d, global_status, first_candle_label,
                                                 trail_tiers, extended_target_r)

        if trade:
            trades.append(trade)

    return {
        "symbol": symbol,
        "symbolLabel": SYMBOL_LABELS.get(symbol, symbol),
        "trades": trades,
        "stats": _backtest_stats(trades),
        "daysAnalyzed": len(trading_dates) - 1,
        "setupsIdentified": setups_identified,
        "dataSource": data_source,
        "entryRangeZone": {"min": ENTRY_RANGE_MIN, "max": ENTRY_RANGE_MAX},
        "stopLossBufferPct": stop_buffer,
        "trailing": {"tiers": trail_tiers, "extendedTargetR": extended_target_r},
        "historyRange": {
            "from": intraday.index[0].strftime('%Y-%m-%d'),
            "to": intraday.index[-1].strftime('%Y-%m-%d')
        }
    }

# ---------------------------------------------------------------------------
# UPDATED: Multi-Tier Trailing Simulation Logic
# ---------------------------------------------------------------------------
def _simulate_trade(day_candles, entry_time, signal, entry, stop, target, risk, trade_date,
                     global_status, first_candle_label,
                     trail_tiers=None, extended_target_r=EXTENDED_TARGET_R):
    """
    Walk forward through the rest of the day's candles to find the exit.

    Multi-tier trailing stop loss:
      - > 1.5R -> Stop moved to Entry (0R)
      - > 2.0R -> Stop moved to +0.5R
      - > 3.0R -> Stop moved to +1.5R
      - > 4.0R -> Stop moved to +3.0R
    """
    if trail_tiers is None:
        trail_tiers = TRAIL_TIERS

    after_entry = day_candles[day_candles.index > entry_time]
    exit_price, exit_reason, exit_time = None, 'EOD_CLOSE', None
    current_stop, current_target = stop, target
    trailed = False

    for t, row in after_entry.iterrows():
        high, low = float(row['High']), float(row['Low'])

        if signal == 'BUY':
            if low <= current_stop:
                exit_price, exit_reason, exit_time = current_stop, ('TRAILED_STOP' if trailed else 'STOPPED_OUT'), t
                break
            if high >= current_target:
                exit_price, exit_reason, exit_time = current_target, 'TARGET_HIT', t
                break

            if risk > 0:
                current_r = (high - entry) / risk
                for tier in sorted(trail_tiers, key=lambda x: x["trigger"], reverse=True):
                    if current_r > tier["trigger"]:
                        new_stop = entry + tier["lock"] * risk
                        if new_stop > current_stop:
                            current_stop = new_stop
                            current_target = entry + extended_target_r * risk
                            trailed = True
                        break

        else:  # SELL
            if high >= current_stop:
                exit_price, exit_reason, exit_time = current_stop, ('TRAILED_STOP' if trailed else 'STOPPED_OUT'), t
                break
            if low <= current_target:
                exit_price, exit_reason, exit_time = current_target, 'TARGET_HIT', t
                break

            if risk > 0:
                current_r = (entry - low) / risk
                for tier in sorted(trail_tiers, key=lambda x: x["trigger"], reverse=True):
                    if current_r > tier["trigger"]:
                        new_stop = entry - tier["lock"] * risk
                        if new_stop < current_stop:
                            current_stop = new_stop
                            current_target = entry - extended_target_r * risk
                            trailed = True
                        break

    if exit_price is None:
        exit_price = float(day_candles.iloc[-1]['Close'])
        exit_time = day_candles.index[-1]

    pnl = (exit_price - entry) if signal == 'BUY' else (entry - exit_price)
    achieved_rr = round(pnl / risk, 2) if risk else 0.0

    return {
        "date": trade_date.strftime('%Y-%m-%d'),
        "signal": signal,
        "globalStatus": global_status,
        "firstCandle": first_candle_label,
        "entryTime": entry_time.strftime('%H:%M'),
        "exitTime": exit_time.strftime('%H:%M') if exit_time is not None else None,
        "entry": round(entry, 2),
        "stopLoss": round(current_stop, 2),
        "target": round(current_target, 2),
        "exitPrice": round(exit_price, 2),
        "riskReward": achieved_rr,
        "status": exit_reason,
        "pnlPoints": round(pnl, 2),
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
    rr_values = [t['riskReward'] for t in trades if isinstance(t.get('riskReward'), (int, float))]
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

def _compute_journal_stats(trades):
    closed_statuses = ('TARGET_HIT', 'STOPPED_OUT', 'CLOSED_SIGNAL_CHANGE')
    closed = [t for t in trades if t['status'] in closed_statuses and t.get('pnlPoints') is not None]
    wins = [t for t in closed if t['pnlPoints'] > 0]
    losses = [t for t in closed if t['pnlPoints'] <= 0]
    open_trades = [t for t in trades if t['status'] == 'OPEN']
    rr_values = [t['riskReward'] for t in trades if isinstance(t.get('riskReward'), (int, float))]

    return {
        'totalSignals': len(trades),
        'openTrades': len(open_trades),
        'closedTrades': len(closed),
        'wins': len(wins),
        'losses': len(losses),
        'winRate': round((len(wins) / len(closed)) * 100, 1) if closed else 0.0,
        'avgRiskReward': round(sum(rr_values) / len(rr_values), 2) if rr_values else 0.0,
        'totalPnlPoints': round(sum(t['pnlPoints'] for t in closed), 2) if closed else 0.0
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
    except (TypeError, AttributeError) as e:
        logger.warning(f"Could not normalize index to IST: {e}")
    return data

def _fetch_yahoo_chart_direct(symbol, period, interval):
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

MIN_STRUCTURE_CANDLES = 21

def detect_market_structure(data, lookback=None, left_bars=10, right_bars=10, min_candles=MIN_STRUCTURE_CANDLES):
    min_required = max(min_candles, left_bars + right_bars + 1)
    empty = {
        "valid": False, "minimumCandles": min_required, "candlesAnalyzed": 0,
        "trend": "neutral", "structure": [], "swingHighs": [], "swingLows": [],
        "lastHighType": None, "lastLowType": None, "score": 0
    }
    if data is None or len(data) < min_required:
        empty["candlesAnalyzed"] = 0 if data is None else len(data)
        return empty

    effective_lookback = lookback if lookback else len(data)
    d = data.tail(max(min_required, effective_lookback)).copy()
    highs, lows = [], []

    for i in range(left_bars, len(d) - right_bars):
        h = float(d["High"].iloc[i])
        l = float(d["Low"].iloc[i])
        hwin = d["High"].iloc[i-left_bars:i+right_bars+1]
        lwin = d["Low"].iloc[i-left_bars:i+right_bars+1]

        if h == float(hwin.max()) and h > float(d["High"].iloc[i-1]):
            highs.append((i, h))
        if l == float(lwin.min()) and l < float(d["Low"].iloc[i-1]):
            lows.append((i, l))

    swing_highs, swing_lows = [], []
    for n, (i, price) in enumerate(highs):
        typ = "SH" if n == 0 else ("HH" if price > highs[n-1][1] else "LH")
        swing_highs.append({
            "index": int(i),
            "timestamp": d.index[i].strftime("%Y-%m-%d %H:%M"),
            "price": round(price, 2),
            "type": typ,
            "confirmed": True
        })

    for n, (i, price) in enumerate(lows):
        typ = "SL" if n == 0 else ("HL" if price > lows[n-1][1] else "LL")
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
    )

    return {
        "valid": True,
        "minimumCandles": min_required,
        "candlesAnalyzed": len(d),
        "trend": trend,
        "structure": structure,
        "swingHighs": swing_highs,
        "swingLows": swing_lows,
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

    supports = _unique_zone_levels(structure.get("swingLows", []), price, "support", tolerance)
    resistances = _unique_zone_levels(structure.get("swingHighs", []), price, "resistance", tolerance)

    return (
        [{"level": f"MS-S{i+1}", "value": round(v, 2), "type": "Market Structure", "source": "HL/LL confirmed swing low"} for i, v in enumerate(supports)],
        [{"level": f"MS-R{i+1}", "value": round(v, 2), "type": "Market Structure", "source": "HH/LH confirmed swing high"} for i, v in enumerate(resistances)]
    )

def generate_trade_signal(data, prediction, cpr, support, resistance, market_structure):
    hold = {
        "signal": "HOLD", "entry": None, "stopLoss": None, "target": None,
        "riskReward": 0, "confidence": 0,
        "reason": "Waiting for aligned prediction, structure, entry and reward"
    }

    if data is None or len(data) < 21:
        hold["reason"] = "Waiting for at least 21 candles"
        return hold

    df = data.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < 21:
        hold["reason"] = "Insufficient valid candles"
        return hold

    last = df.iloc[-1]
    prev = df.iloc[-2]
    entry = float(last["Close"])
    prediction = prediction or {}
    direction = str(prediction.get("direction", prediction.get("prediction", "neutral"))).lower()
    confidence = float(prediction.get("confidence", 50) or 50)

    structure = market_structure or {}
    trend = str(structure.get("trend", "neutral")).lower()
    last_hl = structure.get("lastHL")
    last_lh = structure.get("lastLH")

    ranges = (df["High"] - df["Low"]).tail(min(14, len(df)))
    atr = float(ranges.mean()) if len(ranges) else 0.0
    atr = max(atr, entry * 0.0008, 1.0)

    def values(levels):
        out = []
        for x in levels or []:
            if isinstance(x, dict):
                x = x.get("value", x.get("price", x.get("level")))
            try:
                v = float(x)
                if v > 0:
                    out.append(v)
            except (TypeError, ValueError):
                pass
        return sorted(set(out))

    support_levels = values(support)
    resistance_levels = values(resistance)

    def cpr_num(key):
        try:
            return float(cpr.get(key))
        except (AttributeError, TypeError, ValueError):
            return None

    tc, pivot, bc = cpr_num("tc"), cpr_num("pivot"), cpr_num("bc")
    proximity = max(atr * 0.40, entry * 0.0010)

    hl_breakout = False
    lh_breakdown = False
    hl_price = lh_price = None

    if isinstance(last_hl, dict):
        try:
            hl_idx = int(last_hl.get("index"))
            if 0 <= hl_idx < len(df):
                hl_high = float(df["High"].iloc[hl_idx])
                hl_price = float(last_hl.get("price", df["Low"].iloc[hl_idx]))
                hl_breakout = entry > hl_high and float(prev["Close"]) <= hl_high
        except (TypeError, ValueError, IndexError):
            pass

    if isinstance(last_lh, dict):
        try:
            lh_idx = int(last_lh.get("index"))
            if 0 <= lh_idx < len(df):
                lh_low = float(df["Low"].iloc[lh_idx])
                lh_price = float(last_lh.get("price", df["High"].iloc[lh_idx]))
                lh_breakdown = entry < lh_low and float(prev["Close"]) >= lh_low
        except (TypeError, ValueError, IndexError):
            pass

    near_tc = tc is not None and abs(entry - tc) <= proximity
    near_bc = bc is not None and abs(entry - bc) <= proximity
    bullish_cpr = tc is not None and entry >= tc - proximity
    bearish_cpr = bc is not None and entry <= bc + proximity

    bullish = direction in ("bullish", "buy", "up") and trend == "bullish"
    bearish = direction in ("bearish", "sell", "down") and trend == "bearish"

    if bullish and (near_tc or bullish_cpr or hl_breakout):
        entry_reason = "HL breakout" if hl_breakout else "near/above CPR TC"
        stop_candidates = [v for v in support_levels if v < entry]
        if hl_price is not None and hl_price < entry:
            stop_candidates.append(hl_price)
        structural_stop = max(stop_candidates) if stop_candidates else entry - atr * 1.2
        stop = min(structural_stop - atr * 0.15, entry - atr * 0.60)

        risk = entry - stop
        if risk <= 0:
            return hold

        min_target = entry + risk * 2.0
        higher_resistance = [v for v in resistance_levels if v >= min_target]
        target = higher_resistance[0] if higher_resistance else entry + risk * 2.0
        rr = (target - entry) / risk

        if rr < 2.0:
            return hold

        setup_bonus = 10 if hl_breakout else 5
        return {
            "signal": "BUY",
            "entry": round(entry, 2),
            "stopLoss": round(stop, 2),
            "target": round(target, 2),
            "risk": round(risk, 2),
            "reward": round(target - entry, 2),
            "riskReward": round(rr, 2),
            "confidence": round(min(95, confidence + setup_bonus), 1),
            "reason": f"Bullish prediction + HH/HL + {entry_reason}; target selected at >= 1:{round(rr,2)}",
            "entryZone": "CPR TC / HL confirmation",
            "conditions": {
                "prediction": direction, "structure": trend,
                "nearTC": near_tc, "hlBreakout": hl_breakout
            }
        }

    if bearish and (near_bc or bearish_cpr or lh_breakdown):
        entry_reason = "LH breakdown" if lh_breakdown else "near/below CPR BC"
        stop_candidates = [v for v in resistance_levels if v > entry]
        if lh_price is not None and lh_price > entry:
            stop_candidates.append(lh_price)
        structural_stop = min(stop_candidates) if stop_candidates else entry + atr * 1.2
        stop = max(structural_stop + atr * 0.15, entry + atr * 0.60)

        risk = stop - entry
        if risk <= 0:
            return hold

        min_target = entry - risk * 2.0
        lower_support = sorted([v for v in support_levels if v <= min_target], reverse=True)
        target = lower_support[0] if lower_support else entry - risk * 2.0
        rr = (entry - target) / risk

        if rr < 2.0:
            return hold

        setup_bonus = 10 if lh_breakdown else 5
        return {
            "signal": "SELL",
            "entry": round(entry, 2),
            "stopLoss": round(stop, 2),
            "target": round(target, 2),
            "risk": round(risk, 2),
            "reward": round(entry - target, 2),
            "riskReward": round(rr, 2),
            "confidence": round(min(95, confidence + setup_bonus), 1),
            "reason": f"Bearish prediction + LH/LL + {entry_reason}; target selected at >= 1:{round(rr,2)}",
            "entryZone": "CPR BC / LH confirmation",
            "conditions": {
                "prediction": direction, "structure": trend,
                "nearBC": near_bc, "lhBreakdown": lh_breakdown
            }
        }

    hold["confidence"] = round(confidence, 1)
    hold["reason"] = f"HOLD: prediction={direction}, structure={trend}; waiting for high-quality entry near CPR or confirmed breakout"
    return hold

def generate_candlestick_data(data, max_candles=None):
    if data is None or len(data) == 0:
        return []
    d = data.tail(max_candles) if max_candles else data
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

def calculate_volume_analysis(data):
    try:
        if data is None or len(data) == 0 or 'Volume' not in data.columns:
            return {'current_volume': 0, 'avg_volume': 0, 'volume_ratio': 0.0, 'volume_trend': 'unknown'}

        volume = data['Volume'].fillna(0)
        current_volume = float(volume.iloc[-1])
        lookback = min(20, len(volume))
        avg_volume = float(volume.tail(lookback).mean()) if lookback > 0 else 0.0
        volume_ratio = float(round(current_volume / avg_volume, 2)) if avg_volume > 0 else 0.0

        if avg_volume <= 0:
            trend = 'unknown'
        elif volume_ratio >= 1.5:
            trend = 'surging'
        elif volume_ratio >= 1.1:
            trend = 'increasing'
        elif volume_ratio <= 0.5:
            trend = 'very low'
        elif volume_ratio <= 0.9:
            trend = 'decreasing'
        else:
            trend = 'normal'

        return {
            'current_volume': int(current_volume),
            'avg_volume': int(round(avg_volume)),
            'volume_ratio': volume_ratio,
            'volume_trend': trend
        }
    except Exception as e:
        logger.error(f"Error calculating volume analysis: {e}")
        return {'current_volume': 0, 'avg_volume': 0, 'volume_ratio': 0.0, 'volume_trend': 'unknown'}

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

        try:
            record_signal_for_journal(symbol, timeframe, trade_signal, data)
        except Exception as journal_err:
            logger.error(f"Signal journal update failed for {symbol}: {journal_err}")

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

        nifty_data_raw = fetch_market_data('^NSEI', timeframe)
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

        def unavailable_market(symbol, reason):
            return {
                'available': False,
                'symbol': symbol,
                'current': None,
                'open': None,
                'high': None,
                'low': None,
                'change': None,
                'changePercent': None,
                'cpr': {'pivot': None, 'tc': None, 'bc': None, 'basis': 'N/A', 'period_label': 'N/A'},
                'support': [],
                'resistance': [],
                'sr_info': {'basis': 'N/A', 'period_label': 'N/A'},
                'marketStructure': {
                    'valid': False, 'enoughCandles': False,
                    'minimumCandles': 20, 'trend': 'neutral', 'structure': [],
                    'swingHighs': [], 'swingLows': []
                },
                'tradeSignal': {
                    'signal': 'HOLD',
                    'entry': None, 'stopLoss': None, 'target': None,
                    'risk': None, 'reward': None, 'riskReward': 0,
                    'confidence': 0,
                    'reason': reason
                },
                'chartOverlays': {'cpr': [], 'support': [], 'resistance': [], 'structureMarkers': []},
                'candleData': [],
                'volume': {},
                'indicators': {},
                'dataPoints': 0,
                'dataProviderStatus': 'temporarily_unavailable'
            }

        if nifty_data is None:
            nifty_data = unavailable_market('^NSEI', 'Nifty 50 data temporarily unavailable')
        if banknifty_data is None:
            banknifty_data = unavailable_market('^NSEBANK', 'Bank Nifty data temporarily unavailable')

        provider_available = bool(nifty_data.get('candleData') or banknifty_data.get('candleData'))

        return jsonify({
            'timeframe': timeframe,
            'timeframe_label': TIMEFRAMES[timeframe]['label'],
            'prediction': prediction,
            'globalMarkets': global_markets,
            'nifty': nifty_data,
            'bankNifty': banknifty_data,
            'timestamp': now_ist().isoformat(),
            'market_status': get_market_status(),
            'status': 'success' if provider_available else 'degraded',
            'data_provider_status': (
                'available' if provider_available else 'temporarily_unavailable'
            ),
            'nifty_available': bool(nifty_data.get('candleData')),
            'banknifty_available': bool(banknifty_data.get('candleData')),
            'response_version': '3.2'
        })
    except Exception as e:
        logger.exception("Global handler exception")
        return jsonify({
            'error': 'Internal market data processing error',
            'details': str(e),
            'status': 'error'
        }), 500

def get_market_status():
    now = now_ist()
    if now.weekday() >= 5: return 'closed'
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return 'open' if market_open <= now <= market_close else 'closed'

@app.route('/api/signal-log', methods=['GET'])
def get_signal_log():
    try:
        symbol_param = str(request.args.get('symbol', 'all')).strip().lower()
        if symbol_param in ('nifty', 'nifty50', 'nifty 50', '^nsei'):
            symbols = ['^NSEI']
        elif symbol_param in ('banknifty', 'bank_nifty', 'bank nifty', '^nsebank'):
            symbols = ['^NSEBANK']
        else:
            symbols = ['^NSEI', '^NSEBANK']

        try:
            limit = max(1, min(500, int(request.args.get('limit', 100))))
        except (TypeError, ValueError):
            limit = 100

        with SIGNAL_LOG_LOCK:
            combined = [dict(t) for sym in symbols for t in SIGNAL_LOG.get(sym, [])]

        combined.sort(key=lambda t: t['id'], reverse=True)
        combined = combined[:limit]

        return jsonify({
            'trades': combined,
            'stats': _compute_journal_stats(combined),
            'count': len(combined),
            'timestamp': now_ist().isoformat()
        })
    except Exception as e:
        logger.exception("Signal log handler exception")
        return jsonify({'error': 'Failed to load signal log', 'details': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

def _resolve_symbols_param(raw):
    v = str(raw or 'all').strip().lower()
    if v in ('nifty', 'nifty50', 'nifty 50', '^nsei'):
        return ['^NSEI']
    if v in ('banknifty', 'bank_nifty', 'bank nifty', '^nsebank'):
        return ['^NSEBANK']
    if v in ('sensex', 'bsesn', '^bsesn', 'bse', 'bse sensex'):
        return ['^BSESN']
    return ['^NSEI', '^NSEBANK', '^BSESN']

@app.route('/api/history/sync', methods=['GET', 'POST'])
def sync_history():
    if not github_configured():
        return jsonify({
            'error': 'GitHub storage is not configured on the server',
            'details': 'Set GITHUB_TOKEN, GITHUB_REPO (and optionally GITHUB_BRANCH, GITHUB_DATA_DIR) as environment variables.'
        }), 400

    symbols = _resolve_symbols_param(request.args.get('symbol', 'all'))
    results = [sync_5m_history_to_github(sym) for sym in symbols]
    return jsonify({'results': results, 'timestamp': now_ist().isoformat()})

@app.route('/api/history/status', methods=['GET'])
def history_status():
    symbols = _resolve_symbols_param(request.args.get('symbol', 'all'))
    results = []
    for sym in symbols:
        if not github_configured():
            results.append({'symbol': sym, 'symbolLabel': SYMBOL_LABELS.get(sym, sym), 'configured': False})
            continue
        df = load_5m_history(sym)
        if df is None or len(df) == 0:
            results.append({'symbol': sym, 'symbolLabel': SYMBOL_LABELS.get(sym, sym), 'configured': True, 'rows': 0})
            continue
        trading_days = pd.Series(df.index.date).nunique()
        results.append({
            'symbol': sym, 'symbolLabel': SYMBOL_LABELS.get(sym, sym), 'configured': True,
            'rows': int(len(df)), 'tradingDays': int(trading_days),
            'from': df.index[0].strftime('%Y-%m-%d'), 'to': df.index[-1].strftime('%Y-%m-%d')
        })
    return jsonify({'symbols': results, 'retentionDays': HISTORY_RETENTION_DAYS, 'timestamp': now_ist().isoformat()})

@app.route('/api/backtest', methods=['GET'])
def get_backtest():
    try:
        symbol = _resolve_symbols_param(request.args.get('symbol', 'nifty'))[0]
        try:
            days = max(1, min(HISTORY_RETENTION_DAYS, int(request.args.get('days', HISTORY_RETENTION_DAYS))))
        except (TypeError, ValueError):
            days = HISTORY_RETENTION_DAYS

        result = run_backtest(symbol, days=days, stop_buffer=STOP_LOSS_BUFFER_PCT,
                              trail_tiers=TRAIL_TIERS, extended_target_r=EXTENDED_TARGET_R)
        result['timestamp'] = now_ist().isoformat()
        return jsonify(result)
    except Exception as e:
        logger.exception("Backtest handler exception")
        return jsonify({'error': 'Backtest failed', 'details': str(e)}), 500

# ---------------------------------------------------------------------------
# NEW: File Download Endpoint
# ---------------------------------------------------------------------------
@app.route('/api/download-app', methods=['GET'])
def download_app():
    """Allows downloading the current source code file."""
    try:
        return send_file(__file__, as_attachment=True, download_name="app.py")
    except Exception as e:
        logger.exception("Download application exception")
        return jsonify({'error': 'Failed to download file', 'details': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({'service': 'Indian Stock Market Predictor', 'version': '2.1.0-updated'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
