# backend/online_inference_yf.py
"""
Online T+N inference based on yfinance data and pretrained XGBoost models.

Important:
- Exports:
    - UX_SOFT_THR (threshold for a "soft" BUY hint, based on prob_up)
    - predict_live(symbol, horizon_days, start=None)
    - predict_live_tplus1(symbol, start=None) — alias for horizon_days=1

If the start parameter is not passed, the module automatically takes
the dynamic range for approximately the last year:
this is sufficient for Z_WIN=180 and EMA200, but does not load unnecessary
history from 2018.

Returns a dict with keys:
    - “asof_time”          — when the forecast was made (UTC, ISO)
    - “pred_date”          — target date T+N (taking into account the crypto/equity calendar)
    - “prob_up_calibrated” — P(up) after calibration (if iso is available)
    - “decision_long”      — final long/flat decision based on v10.1 logic
    - “ux_verdict”         — human-readable verdict (“Buy” / ‘Caution’ / “Neutral”)
    - “horizon_days”       — int

Online inference under v10.1:
1) pull OHLCV from yfinance (start → today),
2) build base features using the same TA pipeline as offline,
3) calculate z-features (z180_*, rz180_*) as in the train script,
4) collect a feature vector for the last date using meta[“features”],
5) run it through the XGBoost classifier/regressor,
6) apply alpha/thr/gate_value/regime_ema from meta,
7) form a UX verdict.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import xgboost as xgb
import json

# ==================== UX CONFIG ====================
# “soft” BUY based on probability
UX_SOFT_THR: float = 0.29
UX_CONF_BAND: float = 0.05  # strong “Buy” / “Caution” zone around 0.5

# ==================== HORIZONS & SYMBOLS ====================
SUPPORTED_HORIZONS = [1, 3, 7]

# mapping internal tickers → yfinance tickers
YF_TICKERS: Dict[str, str] = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "TSLA": "TSLA",
    "AAPL": "AAPL",
}

# calendar for T+N
ASSET_CALENDAR: Dict[str, str] = {
    "BTC": "crypto",
    "ETH": "crypto",
    "TSLA": "equity",
    "AAPL": "equity",
}

# base dir: project root
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

# mapping: horizon_days -> model_version_suffix
HORIZON_MODEL_VERSION: Dict[int, str] = {
    1: "hybrid_xgb_multi_tplus1_classic_gate_v10_1_v11ux",
    3: "hybrid_xgb_multi_tplus3_classic_gate_v10_1_v11ux_v2",
    7: "hybrid_xgb_multi_tplus7_classic_gate_v10_1_v11ux",
}

Z_WIN: int = 180
Z_MINP: int = 90

# in-memory cache: (symbol, horizon) -> {"clf","reg","meta","iso"}
_MODEL_CACHE: dict[tuple[str, int], Dict[str, Any]] = {}


# ==================== DYNAMIC START HELPER ====================

def _default_start_for_live() -> str:
    """
    Динамический старт для онлайн-инференса:
    берём последние 365 дней от текущей даты (UTC).

    Этого достаточно для:
      - z-окна Z_WIN=180 (минимум 90 баров),
      - EMA200 (режим фильтра),
    и при этом не тянем историю с 2018 года.
    """
    today = pd.Timestamp.utcnow().normalize()
    start = today - pd.Timedelta(days=365)
    return start.strftime("%Y-%m-%d")


# ==================== TA HELPERS (как в feature-пайплайне) ====================

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def bollinger(
    close: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    width = (upper - lower) / (ma.replace(0, np.nan))
    pb = (close - lower) / (upper - lower)
    return ma, upper, lower, width, pb


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return true_range(high, low, close).rolling(window, min_periods=window).mean()


def cyclical_time_features(dt_index: pd.Series) -> pd.DataFrame:
    dow = dt_index.dt.weekday
    month = dt_index.dt.month
    return pd.DataFrame(
        {
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "mon_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "mon_cos": np.cos(2 * np.pi * (month - 1) / 12),
        },
        index=dt_index.index,
    )


def safe_log1p(s: pd.Series) -> pd.Series:
    return np.log1p(s.astype(float))


# ==================== DOWNLOAD & STANDARDIZE OHLC ====================

def _download_ohlc(symbol: str, start: str = "2018-01-01") -> pd.DataFrame:
    """
    Тянем daily OHLCV из yfinance, приводим к колонкам:
    date, open, high, low, close, volume, adj_close (если есть).
    """
    sym_upper = symbol.upper()
    yf_symbol = YF_TICKERS.get(sym_upper, sym_upper)

    try:
        df = yf.download(
            yf_symbol,
            start=start,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"{sym_upper}: error while calling yfinance "
            f"(yf_symbol={yf_symbol}, start={start}): {e}"
        )

    if df is None or df.empty:
        raise RuntimeError(
            f"{sym_upper}: yfinance returned an empty dataframe "
            f"(yf_symbol={yf_symbol}, start={start})"
        )

    # 1) Flatten MultiIndex, если вдруг он есть
    if isinstance(df.columns, pd.MultiIndex):
        canonical = {
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "open",
            "high",
            "low",
            "close",
            "adj close",
            "volume",
        }
        new_cols = []
        for col in df.columns:
            chosen = None
            for lvl in col:
                s = str(lvl)
                if s in canonical or s.capitalize() in canonical:
                    chosen = s
                    break
            if chosen is None:
                chosen = str(col[-1])
            new_cols.append(chosen)
        df.columns = new_cols
    else:
        df.columns = [str(c) for c in df.columns]

    df = df.reset_index()

    possible_date_cols = ["Date", "date", "Datetime", "datetime", "index"]
    date_col = None
    for c in possible_date_cols:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        # fallback
        date_col = df.columns[0]

    if date_col != "date":
        df = df.rename(columns={date_col: "date"})

    # 3) rename OHLCV
    rename_map = {}
    for src, tgt in [
        ("Open", "open"),
        ("High", "high"),
        ("Low", "low"),
        ("Close", "close"),
        ("Adj Close", "adj_close"),
        ("Volume", "volume"),
    ]:
        if src in df.columns:
            rename_map[src] = tgt
        if src.lower() in df.columns:
            rename_map[src.lower()] = tgt

    df = df.rename(columns=rename_map)

    if "close" not in df.columns:
        raise RuntimeError(
            f"{sym_upper}: column 'close' not found after normalization. "
            f"Current columns={list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # We only keep what is necessary for features.
    keep_cols = ["date", "open", "high", "low", "close", "volume"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    if df.shape[0] < (Z_WIN + 10):
        # strict guard — enough history is needed for z180
        raise RuntimeError(
            f"{sym_upper}: not enough history from yfinance "
            f"(rows={df.shape[0]}, need at least ~{Z_WIN + 10})"
        )

    return df


# ==================== RAW FEATURE PIPELINE (как build_features) ====================

def _build_raw_features_for_live(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    We build the same set of base features as offline build_features,
    but:
      - we do not create next_close / next_return / y_class,
      - we do not drop the last row.
    The output is a DataFrame with columns:
        close_time, close, open, high, low, volume, ...
        ret_1d, ret_kd, EMA, MACD, Bollinger, ATR, volume_log, cyclical time etc.
    """
    df = df_ohlc.copy()
    df = df.rename(columns={"date": "close_time"})
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)

    # Base returns / lags
    df["ret_1d"] = df["close"].pct_change()
    for k in [2, 3, 5, 10]:
        df[f"ret_{k}d"] = df["close"].pct_change(k)
    for l in [1, 2, 3, 5, 10]:
        df[f"close_lag_{l}"] = df["close"].shift(l)

    # Rolling stats
    for win in [5, 10, 20]:
        df[f"sma_{win}"] = df["close"].rolling(win, min_periods=win).mean()
        df[f"std_{win}"] = df["close"].rolling(win, min_periods=win).std()
        df[f"ret_std_{win}"] = df["ret_1d"].rolling(win, min_periods=win).std()

    # Trend / momentum
    df["ema_12"] = ema(df["close"], 12)
    df["ema_26"] = ema(df["close"], 26)
    df["rsi_14"] = rsi(df["close"], 14)
    macd_line, signal_line, macd_hist = macd(df["close"], 12, 26, 9)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd_line, signal_line, macd_hist

    # Bands / Volatility
    bb_ma, _, _, bb_w, bb_pb = bollinger(df["close"], 20, 2.0)
    df["bb_ma_20"], df["bb_width_20"], df["bb_percent_b_20"] = bb_ma, bb_w, bb_pb
    df["hl2"] = (df["high"] + df["low"]) / 2.0
    df["hl_spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-12)
    df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)

    # Volume transforms
    if "volume" in df.columns:
        df["volume_log"] = safe_log1p(df["volume"])
        for win in [5, 20]:
            vol_mean = df["volume"].rolling(win, min_periods=win).mean()
            vol_std = df["volume"].rolling(win, min_periods=win).std()
            df[f"vol_sma_{win}"] = vol_mean
            df[f"volume_z_{win}"] = (df["volume"] - vol_mean) / (vol_std + 1e-12)

    # Time features
    df = pd.concat([df, cyclical_time_features(df["close_time"])], axis=1)

    # We only remove the “head” with NaNs, since z180 will still burn through another part of the history.
    df_feat = df.dropna().reset_index(drop=True)
    if df_feat.empty:
        raise RuntimeError("No valid rows left after building raw features")
    return df_feat


# ==================== Z-STATIONARIZATION ====================

def _rolling_z(s: pd.Series, win: int = Z_WIN, minp: int = Z_MINP) -> pd.Series:
    mu = s.rolling(win, min_periods=minp).mean().shift(1)
    sd = s.rolling(win, min_periods=minp).std().shift(1)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def _rolling_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _rolling_robust_z(s: pd.Series, win: int = Z_WIN, minp: int = Z_MINP) -> pd.Series:
    med = s.rolling(win, min_periods=minp).median().shift(1)
    mad = s.rolling(win, min_periods=minp).apply(_rolling_mad, raw=True).shift(1)
    denom = 1.4826 * mad.replace(0, np.nan)
    z = (s - med) / denom
    return z.replace([np.inf, -np.inf], np.nan)


def _build_stationary_features_live(
    df_feat: pd.DataFrame,
    regime_ema: int,
) -> pd.DataFrame:
    """
    Input: raw features (like df_feat from build_features to targets).
Output:
close_time, close, ema_200, z180_*, rz180_*
with NaN drop by z-window.
    """
    df = df_feat.copy()
    time_col = "close_time"

    # EMA200
    if regime_ema and "close" in df.columns:
        df["ema_200"] = df["close"].ewm(span=regime_ema, adjust=False).mean()
    else:
        df["ema_200"] = np.nan

    # num_cols
    ban_cols = {
        "y_class",
        "next_return",
        time_col,
        "close",
        "open_time",
        "close_time",
        "ignore",
        "symbol",
    }
    num_cols = [
        c for c in df.columns
        if c not in ban_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    Z = pd.DataFrame(index=df.index, dtype=float)

    for c in num_cols:
        Z[f"z{Z_WIN}_{c}"] = _rolling_z(df[c].astype(float))

    heavy = [
        c
        for c in [
            "volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "volume_log",
            "num_trades_log",
            "taker_buy_base_log",
            "taker_buy_quote_log",
        ]
        if c in df.columns
    ]
    for c in heavy:
        Z[f"rz{Z_WIN}_{c}"] = _rolling_robust_z(df[c].astype(float))

    work = pd.concat(
        [df[[time_col, "close", "ema_200"]], Z],
        axis=1,
    )

    work = work.dropna().reset_index(drop=True)
    if work.empty:
        raise RuntimeError(
            "Not enough history after z-normalization "
            f"(Z_WIN={Z_WIN}, Z_MINP={Z_MINP})"
        )
    return work


# ==================== MODEL LOADING ====================

def _load_bundle(symbol: str, horizon_days: int) -> Dict[str, Any]:
    """
    Загружаем связку:
      - clf: xgb.Booster (классификатор)
      - reg: xgb.Booster (регрессор по амплитуде)
      - meta: dict (alpha, thr, regime_ema, features, ...)
      - iso: IsotonicRegression (если сохранён .iso.joblib)
    """
    key = (symbol.upper(), int(horizon_days))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model_version = HORIZON_MODEL_VERSION.get(horizon_days)
    if model_version is None:
        raise RuntimeError(f"No model_version configured for horizon_days={horizon_days}")

    base_name = f"{key[0]}_{model_version}"
    clf_path = MODELS_DIR / f"{base_name}.clf.json"
    reg_path = MODELS_DIR / f"{base_name}.reg.json"
    meta_path = MODELS_DIR / f"{base_name}.meta.json"
    iso_path = MODELS_DIR / f"{base_name}.iso.joblib"

    if not (clf_path.exists() and reg_path.exists() and meta_path.exists()):
        raise RuntimeError(
            f"Model artifacts not found for {base_name} "
            f"({clf_path.name}, {reg_path.name}, {meta_path.name})"
        )

    clf = xgb.Booster()
    clf.load_model(str(clf_path))

    reg = xgb.Booster()
    reg.load_model(str(reg_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # isotonic
    if iso_path.exists():
        iso = joblib.load(iso_path)
    else:
        iso = None

    bundle = {"clf": clf, "reg": reg, "meta": meta, "iso": iso}
    _MODEL_CACHE[key] = bundle
    return bundle


# ==================== TRADING CALENDAR ====================

def _shift_trading_days(
    dt: pd.Timestamp,
    n: int,
    calendar: str,
) -> pd.Timestamp:
    """
    Crypto: +n календарных дней.
    Equity: +n торговых дней (пропуская выходные).
    """
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")

    d = dt.normalize()

    if n <= 0:
        return d

    if calendar == "crypto":
        # Cryptocurrency is traded 24/7 — just shift it by n calendar days
        return (d + pd.Timedelta(days=n)).normalize()

    # Promotions: we follow the calendar, counting only weekdays
    steps = 0
    while steps < n:
        d = d + pd.Timedelta(days=1)
        if d.weekday() < 5:  # 0=Mon ... 4=Fri
            steps += 1
    return d


# ==================== CORE INFERENCE HELPERS ====================

def _build_live_matrix(
    symbol: str,
    horizon_days: int,
    start: str,
    meta: Dict[str, Any],
) -> Tuple[pd.DataFrame, float, float, pd.Timestamp]:
    """
    Build X_last based on the meta[“features”] feature set for online inference.
    Returns:
        X_last   — DataFrame with one row and columns meta[“features”]
        close    — last close price
        ema200   — last ema_200 (or NaN)
        bar_time — close_time of the last bar
    """
    df_ohlc = _download_ohlc(symbol, start=start)
    df_raw = _build_raw_features_for_live(df_ohlc)

    regime_ema = int(meta.get("regime_ema", 0))
    work = _build_stationary_features_live(df_raw, regime_ema=regime_ema)

    row_last = work.iloc[-1]
    bar_time = pd.to_datetime(row_last["close_time"])
    close_last = float(row_last["close"])
    ema200_last = float(row_last.get("ema_200", np.nan))

    feat_cols_model = list(meta.get("features", []))
    if not feat_cols_model:
        raise RuntimeError("meta['features'] is empty — cannot build X matrix")

    # Collecting X_last: missing features → 0.0 (neutral z-level)
    data = {}
    for col in feat_cols_model:
        if col in work.columns:
            data[col] = [float(row_last[col])]
        else:
            # if the feature cannot be calculated online (e.g., rz180_num_trades),
            # we use 0.0 — this is the “average” z-level.
            data[col] = [0.0]
    X_last = pd.DataFrame(data, index=[0], columns=feat_cols_model)
    return X_last, close_last, ema200_last, bar_time


def _infer_with_bundle(
    bundle: Dict[str, Any],
    X_last: pd.DataFrame,
    close_last: float,
    ema200_last: float,
) -> Tuple[float, float, bool, float]:
    """
    Passing the last line through the bundle (clf, reg, meta, iso):

    Returns:
        prob_up   — calibrated probability of growth (P(up))
        blend     — final score after alpha blending
        decision  — final long/flat decision based on thr + gate + regime threshold
        ret_score — normalized signal amplitude ([-1..1])
    """
    clf: xgb.Booster = bundle["clf"]
    reg: xgb.Booster = bundle["reg"]
    meta: Dict[str, Any] = bundle["meta"]
    iso = bundle.get("iso", None)

    feat_cols = list(meta.get("features", []))
    if not feat_cols:
        raise RuntimeError("meta['features'] is empty")

    alpha = float(meta.get("alpha", 0.0))
    thr = float(meta.get("thr", 0.5))
    regime_ema = int(meta.get("regime_ema", 0))

    # For the ret-regressor:
    # if scale_vol / gate_value are not saved in meta (old models),
    # use safe defaults.
    scale_vol = float(meta.get("scale_vol", 0.02))  # типичный дневной σ ~2%
    gate_value = float(meta.get("gate_value", 0.0))  # 0.0 => gate shut down

    dmat = xgb.DMatrix(
        X_last[feat_cols].to_numpy(dtype=np.float32),
        feature_names=feat_cols,
    )

    # --- classifier: P(up) ---
    proba_raw = clf.predict(dmat)
    if isinstance(proba_raw, (np.ndarray, list)):
        proba_raw = float(proba_raw[0])
    else:
        proba_raw = float(proba_raw)

    if iso is not None:
        proba_arr = iso.transform([proba_raw])
        prob_up = float(proba_arr[0])
    else:
        prob_up = float(proba_raw)

    # --- Regressor: amplitude ---
    ret_pred = reg.predict(dmat)
    if isinstance(ret_pred, (np.ndarray, list)):
        ret_pred = float(ret_pred[0])
    else:
        ret_pred = float(ret_pred)
    scale_vol = max(scale_vol, 1e-6)
    ret_score = float(np.tanh(ret_pred / (3.0 * scale_vol)))  # [-1..1]

    # --- Alpha-blend: proba + magnitude-score ---
    blend = (1.0 - alpha) * prob_up + alpha * (ret_score * 0.5 + 0.5)

    # --- Gate module ret_score ---
    if gate_value > 0.0:
        trade = bool(abs(ret_score) >= gate_value)
    else:
        # if gate_value is not specified in meta, disable gate
        trade = True

    # --- Regime filter по EMA200 ---
    if regime_ema:
        regime_on = bool(close_last > ema200_last)
    else:
        regime_on = True

    decision_long = bool(blend >= thr and trade and regime_on)

    return prob_up, blend, decision_long, ret_score


# ==================== PUBLIC API ====================

def predict_live(
    symbol: str,
    horizon_days: int = 1,
    start: str | None = None,
) -> Dict[str, Any]:
    """
    The main entry point for FastAPI:

        res = predict_live(“BTC”, horizon_days=1)

    If start is not specified, we automatically take the last 365 days
    via _default_start_for_live().

    Returns dict:
        - “asof_time”          — ISO UTC (forecast calculation time)
        - “pred_date”          — ISO UTC (T+N according to the asset calendar)
        - “prob_up_calibrated” — P(up)
        - “decision_long”      — bool
        - “ux_verdict”         — str (ru)
        - “horizon_days”       — int
    """
    sym_upper = symbol.upper()
    if horizon_days not in SUPPORTED_HORIZONS:
        raise RuntimeError(
            f"Online inference: horizon_days={horizon_days} not in {SUPPORTED_HORIZONS}"
        )

    calendar = ASSET_CALENDAR.get(sym_upper, "equity")

    # dynamic start, unless explicitly passed
    if start is None:
        start = _default_start_for_live()

    # 1. Upload the bundle (clf, reg, meta, iso)
    bundle = _load_bundle(sym_upper, horizon_days)
    meta: Dict[str, Any] = bundle["meta"]

    # 2. Building live features and X_last under meta[“features”]
    X_last, close_last, ema200_last, bar_time = _build_live_matrix(
        sym_upper,
        horizon_days=horizon_days,
        start=start,
        meta=meta,
    )

    # 3. Run through models
    prob_up, blend, decision_long, ret_score = _infer_with_bundle(
        bundle=bundle,
        X_last=X_last,
        close_last=close_last,
        ema200_last=ema200_last,
    )

    # 4 Times: asof_time = “now”, pred_date = T+N from the last bar on the asset calendar
    asof_time = pd.Timestamp.utcnow().floor("min")
    pred_date = _shift_trading_days(
        dt=bar_time,
        n=int(horizon_days),
        calendar=calendar,
    )
    pred_date = pred_date.normalize()

    # 5. UX verdict on probability (can be changed to blend if desired)
    if prob_up >= 0.5 + UX_CONF_BAND:
        ux_verdict = "Покупай"
    elif prob_up <= 0.5 - UX_CONF_BAND:
        ux_verdict = "Осторожно"
    else:
        ux_verdict = "Нейтрально"

    return {
        "asof_time": asof_time.isoformat(),
        "pred_date": pred_date.isoformat(),
        "prob_up_calibrated": float(prob_up),
        "decision_long": bool(decision_long),
        "ux_verdict": ux_verdict,
        "horizon_days": int(horizon_days),
    }


def predict_live_tplus1(
    symbol: str,
    start: str | None = None,
) -> Dict[str, Any]:
    """
    Backward-compatible alias: T+1 online forecast.
    If start is not specified, a dynamic range of ~365 days is used.
    """
    return predict_live(symbol, horizon_days=1, start=start)
