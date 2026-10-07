"""
Indian Stock Market Predictor - Multi-Timeframe Backend
Real-time data with multiple timeframe support - Patched for curl_cffi / yfinance cookie crash
Updated with Daily 09:15 IST Global Index Status Snapshot & History Tracker for Backtesting
Includes RCBO (Red Candle Breakout) Backtesting Engine Integration
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

# ---------------------------------------------------------------------------
# Global-market status: retry-safe fetch + frozen per-date store
# ---------------------------------------------------------------------------
GLOBAL_STATUS_FILE = "GLOBAL_STATUS.json"
GLOBAL_FETCH_RETRIES = 3
try:
    MIN_GLOBAL_SYMBOLS = int(_clean_env("MIN_GLOBAL_SYMBOLS", default="8"))
except ValueError:
    MIN_GLOBAL_SYMBOLS = 8
GLOBAL_FREEZE_HOUR_IST = 4          # freeze a day only after 04:00 IST next morning
GLOBAL_ADMIN_KEY = _clean_env("ADMIN_KEY", default=None) or None
_VALID_GLOBAL_STATUS = ('bullish', 'bearish', 'neutral')
_GLOBAL_STORE_LOCK = threading.Lock()
_GLOBAL_STORE_LOCAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), GLOBAL_STATUS_FILE)

def fetch_global_daily_history_checked(period='400d'):
    """Returns (data_by_symbol, expected_count, failed_symbols). Each symbol is retried."""
    out, failed, expected = {}, [], 0
    for region, indices in GLOBAL_INDICES.items():
        for sym in indices:
            expected += 1
            df, last_err = None, None
            for attempt in range(1, GLOBAL_FETCH_RETRIES + 1):
                try:
                    df = _normalize_yf_data(get_ticker(sym).history(period=period, interval='1d', timeout=15))
                    if df is not None and len(df) >= 2:
                        break
                    last_err = "empty/short response"
                    df = None
                except Exception as e:
                    last_err, df = str(e), None
                time.sleep(0.5 * attempt)
            if df is not None:
                out[sym] = df
            else:
                failed.append(sym)
                logger.warning(f"Global daily history FAILED for {sym} after {GLOBAL_FETCH_RETRIES} tries: {last_err}")
    logger.info(f"Global daily history: loaded {len(out)}/{expected} indices; failed={failed}")
    return out, expected, failed

def fetch_global_daily_history(period='400d'):
    return fetch_global_daily_history_checked(period)[0]

def historical_global_status_detail(global_daily, target_date):
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
        return {'status': 'neutral', 'positive': 0, 'total': 0}
    ratio = positive / total
    status = 'bullish' if ratio > 0.6 else ('bearish' if ratio < 0.4 else 'neutral')
    return {'status': status, 'positive': positive, 'total': total}

def historical_global_status(global_daily, target_date):
    return historical_global_status_detail(global_daily, target_date)['status']

def _global_store_github_path():
    segments = [s.strip().strip('/') for s in (GITHUB_DATA_DIR, GLOBAL_STATUS_FILE)]
    return "/".join(s for s in segments if s)

def load_global_status_store():
    store, sha = {}, None
    if github_configured():
        content, sha = github_get_file(_global_store_github_path())
        if content:
            try:
                store = json.loads(content)
            except ValueError:
                logger.warning("GLOBAL_STATUS.json on GitHub is not valid JSON; ignoring it")
    if not store and os.path.exists(_GLOBAL_STORE_LOCAL_PATH):
        try:
            with open(_GLOBAL_STORE_LOCAL_PATH, 'r') as fh:
                store = json.load(fh)
        except (OSError, ValueError) as e:
            logger.warning(f"Local global-status store unreadable: {e}")
    return (store if isinstance(store, dict) else {}), sha

def save_global_status_store(store, sha=None):
    payload = json.dumps(store, indent=2, sort_keys=True)
    try:
        with open(_GLOBAL_STORE_LOCAL_PATH, 'w') as fh:
            fh.write(payload)
    except OSError as e:
        logger.warning(f"Could not write local global-status store: {e}")
    if github_configured():
        try:
            if sha is None:
                _, sha = github_get_file(_global_store_github_path())
            github_put_file(_global_store_github_path(), payload,
                            f"Update frozen global statuses ({len(store)} days)", sha=sha)
        except RuntimeError as e:
            logger.warning(f"Could not save global-status store to GitHub: {e}")

def _can_freeze_global_status(d):
    freeze_at = datetime(d.year, d.month, d.day, tzinfo=IST) + timedelta(days=1, hours=GLOBAL_FREEZE_HOUR_IST)
    return now_ist() >= freeze_at

def resolve_global_status(global_daily, data_ok, d, store):
    """Returns (status, source, changed). source: frozen | live | unknown."""
    key = d.strftime('%Y-%m-%d')
    saved = store.get(key)
    if isinstance(saved, dict) and saved.get('status') in _VALID_GLOBAL_STATUS:
        return saved['status'], 'frozen', False
    if not data_ok:
        return 'unknown', 'unknown', False
    detail = historical_global_status_detail(global_daily, d)
    if _can_freeze_global_status(d):
        store[key] = {
            'status': detail['status'], 'positive': detail['positive'], 'total': detail['total'],
            'source': 'auto', 'frozenAt': now_ist().isoformat()
        }
        return detail['status'], 'frozen', True
    return detail['status'], 'live', False

# ---------------------------------------------------------------------------
# Daily 09:15 IST global-status snapshot -> GitHub (market_data/GLOBAL_STATUS.json)
# ---------------------------------------------------------------------------
SNAPSHOT_HOUR, SNAPSHOT_MINUTE = 9, 15                  # Triggered at 09:15 IST
SNAPSHOT_CUTOFF_HOUR, SNAPSHOT_CUTOFF_MINUTE = 15, 30   # Stop retrying after market close
SNAPSHOT_ON_TIME_MINUTES = 10                            # Captured within 10 min of 09:15 IST
SNAPSHOT_RETRY_SECONDS = 300
SCHEDULER_ENABLED = _clean_env("ENABLE_SCHEDULER", default="1").lower() not in ("0", "false", "no", "off")
_scheduler_started = False

def _global_index_details(global_daily, target_date):
    """Fetches real-time intraday data for live snapshots; falls back to history if needed."""
    names = {sym: name for region in GLOBAL_INDICES.values() for sym, name in region.items()}
    today_str = target_date.strftime('%Y-%m-%d')
    details = []

    for sym, name in names.items():
        try:
            cur, prev = None, None
            try:
                t = get_ticker(sym)
                cur = t.fast_info.last_price
                prev = t.fast_info.previous_close
            except Exception as e:
                logger.warning(f"Live ticker fast_info failed for {sym}: {e}")

            if cur is None or prev is None:
                if sym in global_daily:
                    df = global_daily[sym]
                    asof = df[df.index.date <= target_date]
                    if len(asof) >= 2:
                        cur = float(asof['Close'].iloc[-1])
                        prev = float(asof['Close'].iloc[-2])

            if cur is not None and prev is not None and prev > 0:
                change_pct = round(((cur - prev) / prev) * 100, 2)
                details.append({
                    'symbol': sym,
                    'name': name,
                    'barDate': today_str,
                    'close': round(cur, 2),
                    'prevClose': round(prev, 2),
                    'changePct': change_pct
                })
        except Exception as err:
            logger.warning(f"Could not compute index detail for {sym}: {err}")
            continue

    return details

def take_global_snapshot(force=False):
    now = now_ist()
    today = now.date()
    key = today.strftime('%Y-%m-%d')

    if today.weekday() >= 5 and not force:
        return {'status': 'skipped_weekend', 'date': key}

    with _GLOBAL_STORE_LOCK:
        store, _ = load_global_status_store()
    if key in store and not force:
        return {'status': 'already_exists', 'date': key, 'entry': store[key]}

    global_daily, expected, failed = fetch_global_daily_history_checked()
    indices_details = _global_index_details(global_daily, today)
    
    if len(indices_details) < MIN_GLOBAL_SYMBOLS:
        return {'status': 'failed', 'date': key, 'reason': 'too few global indices loaded',
                'indicesLoaded': len(indices_details), 'indicesExpected': expected,
                'minRequired': MIN_GLOBAL_SYMBOLS, 'failed': failed}

    positive_count = sum(1 for idx in indices_details if idx['changePct'] > 0)
    total_count = len(indices_details)
    ratio = positive_count / total_count if total_count > 0 else 0.5
    status = 'bullish' if ratio > 0.6 else ('bearish' if ratio < 0.4 else 'neutral')

    target = now.replace(hour=SNAPSHOT_HOUR, minute=SNAPSHOT_MINUTE, second=0, microsecond=0)
    late = now > target + timedelta(minutes=SNAPSHOT_ON_TIME_MINUTES)
    entry = {
        'status': status,
        'positive': positive_count,
        'total': total_count,
        'source': 'morning_snapshot_0915',
        'capturedAt': now.isoformat(),
        'late': bool(late),
        'indicesFailed': failed,
        'indices': indices_details
    }

    with _GLOBAL_STORE_LOCK:
        store, sha = load_global_status_store()
        if key in store and not force:
            return {'status': 'already_exists', 'date': key, 'entry': store[key]}
        store[key] = entry
        save_global_status_store(store, sha)
    logger.info(f"09:15 IST Global snapshot saved for {key}: {entry['status']} ({entry['positive']}/{entry['total']} up)")
    return {'status': 'ok', 'date': key, 'entry': entry}

def _snapshot_scheduler_tick(state):
    n = now_ist()
    if n.weekday() >= 5 or state.get('done') == n.date():
        return None
    start = n.replace(hour=SNAPSHOT_HOUR, minute=SNAPSHOT_MINUTE, second=0, microsecond=0)
    cutoff = n.replace(hour=SNAPSHOT_CUTOFF_HOUR, minute=SNAPSHOT_CUTOFF_MINUTE, second=0, microsecond=0)
    if not (start <= n < cutoff):
        return None
    if time.time() - state.get('last_try', 0) < SNAPSHOT_RETRY_SECONDS:
        return None
    state['last_try'] = time.time()
    res = take_global_snapshot()
    if res['status'] in ('ok', 'already_exists', 'skipped_weekend'):
        state['done'] = n.date()
    else:
        logger.warning(f"Global snapshot attempt failed, will retry in {SNAPSHOT_RETRY_SECONDS}s: {res}")
    return res

def _snapshot_scheduler_loop():
    state = {}
    logger.info(f"Global snapshot scheduler running (daily {SNAPSHOT_HOUR:02d}:{SNAPSHOT_MINUTE:02d} IST, Mon-Fri)")
    while True:
        try:
            _snapshot_scheduler_tick(state)
        except Exception:
            logger.exception("Global snapshot scheduler error")
        time.sleep(20)

def start_snapshot_scheduler():
    global _scheduler_started
    if _scheduler_started or not SCHEDULER_ENABLED:
        return
    _scheduler_started = True
    threading.Thread(target=_snapshot_scheduler_loop, name="global-snapshot", daemon=True).start()

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
STOP_LOSS_BUFFER_PCT = 0.1

TRAIL_TIERS = [
    {"trigger": 1.5, "lock": 0.0},
    {"trigger": 2.0, "lock": 0.5},
    {"trigger": 3.0, "lock": 1.5},
    {"trigger": 4.0, "lock": 3.0},
]
EXTENDED_TARGET_R = 10.0

def _choose_stop_loss(signal, entry, entry_row, bc, tc, buffer_pct=STOP_LOSS_BUFFER_PCT):
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

def _simulate_trade(day_candles, entry_time, signal, entry, stop, target, risk, trade_date,
                     global_status, first_candle_label,
                     trail_tiers=None, extended_target_r=EXTENDED_TARGET_R):
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

        else:
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

# ---------------------------------------------------------------------------
# Red Candle Breakout (RCBO) Specific Backtesting Logic
# ---------------------------------------------------------------------------
def run_rcbo_backtest(symbol, days=60, min_rr=2.0, max_candle_pct=0.8, filter_cpr=True):
    intraday = load_5m_history(symbol)
    data_source = 'github_archive'
    if intraday is None or len(intraday) == 0:
        intraday = fetch_5m_history_chunked(symbol, days=min(days, 60))
        data_source = 'live_fallback_max_60d'

    if intraday is None or len(intraday) == 0:
        return {
            "symbol": symbol, "trades": [], "stats": _backtest_stats([]),
            "daysAnalyzed": 0, "dataSource": "unavailable"
        }

    cutoff = now_ist() - timedelta(days=days)
    intraday = intraday[intraday.index >= cutoff]
    if len(intraday) == 0:
        return {
            "symbol": symbol, "trades": [], "stats": _backtest_stats([]),
            "daysAnalyzed": 0, "dataSource": data_source
        }

    daily_ohlc = _daily_ohlc_from_5m(intraday)
    trades = []
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

        # Central Pivot Range Calculation
        pivot = (pdh + pdl + pdc) / 3.0
        bc = (pdh + pdl) / 2.0
        tc = (pivot - bc) + pivot
        cpr_bottom, cpr_top = min(bc, tc), max(bc, tc)

        day_candles = intraday[intraday.index.date == d]
        if len(day_candles) < 2:
            continue

        # 1. Trigger: First 5-minute candle must be bearish (Close < Open)
        first_candle = day_candles.iloc[0]
        first_open = float(first_candle['Open'])
        first_close = float(first_candle['Close'])
        first_high = float(first_candle['High'])
        first_low = float(first_candle['Low'])

        if not (first_close < first_open):
            continue

        rest_candles = day_candles.iloc[1:]

        # 2. Entry Condition: Subsequent candle closes above first red candle's high
        breakout_candidates = rest_candles[rest_candles['Close'] > first_high]
        if breakout_candidates.empty:
            continue

        entry_row = breakout_candidates.iloc[0]
        entry_time = breakout_candidates.index[0]
        entry_price = float(entry_row['Close'])
        entry_candle_low = float(entry_row['Low'])
        entry_candle_high = float(entry_row['High'])

        # 3. Filter: CPR Resistance Filter
        if filter_cpr:
            if cpr_bottom <= entry_price <= cpr_top or (entry_price < cpr_bottom and first_high >= cpr_bottom):
                continue

        # 4. Stop-Loss Strategy: Entry candle low vs. Previous Day High (PDH) if oversized
        candle_pct = ((entry_candle_high - entry_candle_low) / entry_price) * 100.0
        if candle_pct > float(max_candle_pct):
            stop_loss = pdh if pdh < entry_price else entry_candle_low
            sl_type = 'PDH Fallback'
        else:
            stop_loss = entry_candle_low
            sl_type = 'Entry Candle Low'

        risk = entry_price - stop_loss
        if risk <= 0:
            continue

        target = round(entry_price + risk * float(min_rr), 2)

        # 5. Execute Simulation with trailing stop loss support
        trade = _simulate_trade(
            day_candles, entry_time, 'BUY', entry_price, stop_loss, target, risk, d,
            global_status='RCBO_TRIGGERED', first_candle_label='red',
            trail_tiers=TRAIL_TIERS, extended_target_r=EXTENDED_TARGET_R
        )
        trade['slType'] = sl_type
        trades.append(trade)

    return {
        "symbol": symbol,
        "symbolLabel": SYMBOL_LABELS.get(symbol, symbol),
        "trades": trades,
        "stats": _backtest_stats(trades),
        "daysAnalyzed": len(trading_dates) - 1,
        "dataSource": data_source
    }

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route('/api/backtest/rcbo', methods=['POST'])
def rcbo_backtest_endpoint():
    try:
        body = request.get_json(silent=True) or {}
        symbol = str(body.get('symbol', '^NSEI'))
        days = int(body.get('days', 60))
        min_rr = float(body.get('min_rr', 2.0))
        max_candle_pct = float(body.get('max_candle_pct', 0.8))
        filter_cpr = str(body.get('filter_cpr', 'true')).lower() == 'true'

        result = run_rcbo_backtest(
            symbol=symbol,
            days=days,
            min_rr=min_rr,
            max_candle_pct=max_candle_pct,
            filter_cpr=filter_cpr
        )
        return jsonify(result)
    except Exception as e:
        logger.exception("RCBO Backtest endpoint exception")
        return jsonify({'error': 'RCBO Backtest failed', 'details': str(e)}), 500

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
            data = ticker.history(period=config['period'], interval=config['interval'], timeout=15, raise_errors=False)
            data = _normalize_yf_data(data)
            if data is not None and not data.empty:
                _MARKET_CACHE[cache_key] = {"timestamp": time.time(), "data": data.copy()}
                return data
        except Exception as e:
            logger.warning(f"yfinance history failed for {symbol}: {e}")

        try:
            data = yf.download(tickers=symbol, period=config['period'], interval=config['interval'], progress=False, threads=False, timeout=15, auto_adjust=False)
            data = _normalize_yf_data(data)
            if data is not None and not data.empty:
                _MARKET_CACHE[cache_key] = {"timestamp": time.time(), "data": data.copy()}
                return data
        except Exception as e:
            logger.warning(f"yfinance download failed for {symbol}: {e}")

        data = _fetch_yahoo_chart_direct(symbol, config['period'], config['interval'])
        if data is not None and not data.empty:
            _MARKET_CACHE[cache_key] = {"timestamp": time.time(), "data": data.copy()}
            return data

        if attempt == 0:
            time.sleep(1)

    logger.error(f"All market-data providers failed for {symbol}")
    return None

def get_market_status():
    now = now_ist()
    if now.weekday() >= 5: return 'closed'
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return 'open' if market_open <= now <= market_close else 'closed'

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/api/download-app', methods=['GET'])
def download_app():
    try:
        return send_file(__file__, as_attachment=True, download_name="app.py")
    except Exception as e:
        logger.exception("Download application exception")
        return jsonify({'error': 'Failed to download file', 'details': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({'service': 'Indian Stock Market Predictor', 'version': '2.4.0-rcbo-integrated'})

start_snapshot_scheduler()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
