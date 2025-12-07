from pathlib import Path
from typing import List, Optional
from datetime import datetime
import json
import os
import sys
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

import certifi
import yfinance as yf
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .online_inference_yf import predict_live, UX_SOFT_THR
from .database import SessionLocal, engine
from .models_db import Base, PredictionRecord, ModelStatsRecord

print("[BOOT] sys.executable =", sys.executable)
print("[BOOT] certifi.where() =", certifi.where())
print("[BOOT] yfinance version =", yf.__version__)
print(f"[BOOT] CURL_CA_BUNDLE = {os.environ.get('CURL_CA_BUNDLE')}")
print(f"[BOOT] SSL_CERT_FILE  = {os.environ.get('SSL_CERT_FILE')}")

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

SUPPORTED_ASSETS = ["BTC", "ETH", "TSLA", "AAPL"]
SUPPORTED_HORIZONS = [1, 3, 7]
DEFAULT_HORIZON = 1

# Horizon mapping -> model version (for meta files)
MODEL_VERSION_TPLUS = {
    1: "hybrid_xgb_multi_tplus1_classic_gate_v10_1_v11ux",
    3: "hybrid_xgb_multi_tplus3_classic_gate_v10_1_v11ux_v2",
    7: "hybrid_xgb_multi_tplus7_classic_gate_v10_1_v11ux",
}

def _load_business_from_meta(symbol: str, horizon_days: int):
    """
    Fallback: if strategy_return / buy_hold_return are not in the database,
    we retrieve them directly from the corresponding meta.json.

    Returns a dict with keys:
      - strategy_return
      - buy_hold_return
      - sharpe
      - win_rate
      - test_period_start (datetime или None)
      - test_period_end   (datetime или None)
    or None if meta is not found or the structure is not suitable.
    """
    version = MODEL_VERSION_TPLUS.get(horizon_days)
    if not version:
        return None

    meta_path = BASE_DIR / "models" / f"{symbol}_{version}.meta.json"
    if not meta_path.exists():
        print(f"[WARN] meta not found for {symbol} T+{horizon_days}: {meta_path}")
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] cannot read meta for {symbol} T+{horizon_days}: {e}")
        return None

    biz = meta.get("business") or {}
    if not isinstance(biz, dict):
        return None

    target = f"T+{horizon_days}"

    def _find_key(prefix: str):
        # 1) First, we try to find a key that clearly contains T+N.
        for k in biz.keys():
            if k.startswith(prefix) and target in k:
                return k
        # 2) if you can't find it, take the first one you find with that prefix
        for k in biz.keys():
            if k.startswith(prefix):
                return k
        return None

    k_strat = _find_key("Strategy Return")
    k_bh = _find_key("Buy&Hold Return")

    strat = biz.get(k_strat) if k_strat else None
    bh = biz.get(k_bh) if k_bh else None

    # test_period: ["2025-04-13 ...", "2025-10-09 ..."]
    test_period = biz.get("test_period") or [None, None]
    test_start, test_end = None, None
    if isinstance(test_period, (list, tuple)) and len(test_period) >= 2:
        s0, s1 = test_period[0], test_period[1]
        try:
            if s0:
                test_start = pd.to_datetime(s0).to_pydatetime()
        except Exception:
            test_start = None
        try:
            if s1:
                test_end = pd.to_datetime(s1).to_pydatetime()
        except Exception:
            test_end = None

    fallback = {
        "strategy_return": strat,
        "buy_hold_return": bh,
        "sharpe": biz.get("Sharpe (all, long-only)"),
        "win_rate": biz.get("Win Rate (executed)"),
        "test_period_start": test_start,
        "test_period_end": test_end,
    }
    return fallback

app = FastAPI(
    title="StockAI Price Prediction API",
    version="1.3.0",
    description=(
        "MVP API: provides T+N forecasts and aggregated metrics on assets "
        "BTC/ETH/TSLA/AAPL из Postgres (predictions, model_stats)."
    ),
)

# === create tables in Postgres if they do not already exist ===
Base.metadata.create_all(bind=engine)

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic models
# =========================

class Prediction(BaseModel):
    symbol: str
    horizon_days: int
    asof_time: datetime
    pred_date: datetime
    up_prob: float
    ux_verdict: str
    ux_soft_buy: bool

class PredictionHistoryPoint(BaseModel):
    symbol: str
    horizon_days: int
    asof_time: datetime
    pred_date: datetime
    up_prob: float
    ux_verdict: str

class AssetSummary(BaseModel):
    symbol: str
    horizon_days: int
    last_up_prob: float
    last_verdict: str
    last_pred_date: datetime

class SignalCard(BaseModel):
    symbol: str
    horizon_days: int
    verdict: str
    up_prob: float
    confidence: str
    price: Optional[float] = None
    change_24h: Optional[float] = None
    strategy_return: Optional[float] = None
    sharpe: Optional[float] = None
    last_update: datetime

class ModelStats(BaseModel):
    symbol: str
    horizon_days: int
    bal_acc: Optional[float] = None
    auc: Optional[float] = None
    strategy_return: Optional[float] = None
    buy_hold_return: Optional[float] = None
    sharpe: Optional[float] = None
    win_rate: Optional[float] = None
    test_period_start: Optional[datetime] = None
    test_period_end: Optional[datetime] = None

# =========================
# DB dependency
# =========================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================
# Internal helpers
# =========================

def _validate_horizon(h: int):
    if h not in SUPPORTED_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"horizon_days={h} Not supported. Acceptable: {SUPPORTED_HORIZONS}",
        )

def _get_last_prediction_from_db(
    db: Session,
    symbol: str,
    horizon_days: int,
) -> PredictionRecord:
    rec = (
        db.query(PredictionRecord)
        .filter(
            PredictionRecord.symbol == symbol,
            PredictionRecord.horizon_days == horizon_days,
        )
        .order_by(PredictionRecord.asof_time.desc())
        .first()
    )
    if rec is None:
        raise HTTPException(
            status_code=404,
            detail=f"No forecasts for {symbol} with a horizon T+{horizon_days}.",
        )
    return rec

def _get_prediction_history_from_db(
    db: Session,
    symbol: str,
    horizon_days: int,
    limit: int,
) -> List[PredictionRecord]:
    q = (
        db.query(PredictionRecord)
        .filter(
            PredictionRecord.symbol == symbol,
            PredictionRecord.horizon_days == horizon_days,
        )
        .order_by(PredictionRecord.asof_time.desc())
    )
    if limit > 0:
        q = q.limit(limit)
    rows = q.all()
    rows.reverse()  # chronological order for charts
    return rows

def _get_model_stats_from_db(
    db: Session,
    symbol: str,
    horizon_days: int,
) -> Optional[ModelStatsRecord]:
    return (
        db.query(ModelStatsRecord)
        .filter(
            ModelStatsRecord.symbol == symbol,
            ModelStatsRecord.horizon_days == horizon_days,
        )
        .first()
    )

def _load_features_last(symbol: str) -> tuple[Optional[float], Optional[float]]:
    """
    Take latest price and 24h change from {symbol}_features.(parquet|csv).
    Returns (price, change_24h_percent) or (None, None).
    """
    pq_path = DATA_DIR / f"{symbol}_features.parquet"
    csv_path = DATA_DIR / f"{symbol}_features.csv"

    df = None

    # 1) Parquet
    if pq_path.exists():
        try:
            df = pd.read_parquet(pq_path)
        except Exception as e:
            print(f"[WARN] parquet read failed for {symbol}: {e}. Fallback to CSV.")

    # 2) CSV
    if df is None and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] csv read failed for {symbol}: {e}")
            return None, None

    # 3) No files
    if df is None:
        print(f"[WARN] no features file for {symbol} ({pq_path.name} / {csv_path.name})")
        return None, None

    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce")
        df = df.sort_values("close_time").reset_index(drop=True)

    if "close" not in df.columns or len(df) < 2:
        return None, None

    last = float(df.iloc[-1]["close"])
    prev = float(df.iloc[-2]["close"])

    if prev == 0:
        return last, None

    change_24h = (last - prev) / prev * 100.0
    return last, change_24h

def _confidence_from_prob(p: float) -> str:
    """
    Simple UX confidence grading based on growth probability.
    """
    if p >= 0.70:
        return "High"
    if p >= 0.55:
        return "Medium"
    return "Low"

def _upsert_online_prediction(
    db: Session,
    symbol: str,
    horizon_days: int,
    asof_time: datetime,
    pred_date: datetime,
    up_prob: float,
    ux_verdict: str,
    ux_soft_buy: bool,
) -> PredictionRecord:
    """
    Upsert logic for online forecasting:
    unique key = (symbol, horizon_days, asof_time).
    """
    existing = (
        db.query(PredictionRecord)
        .filter(
            PredictionRecord.symbol == symbol,
            PredictionRecord.horizon_days == horizon_days,
            PredictionRecord.asof_time == asof_time,
        )
        .first()
    )

    if existing:
        existing.pred_date = pred_date
        existing.up_prob = up_prob
        existing.ux_verdict = ux_verdict
        existing.ux_soft_buy = ux_soft_buy
        db.commit()
        db.refresh(existing)
        return existing

    rec = PredictionRecord(
        symbol=symbol,
        horizon_days=horizon_days,
        asof_time=asof_time,
        pred_date=pred_date,
        up_prob=up_prob,
        ux_verdict=ux_verdict,
        ux_soft_buy=ux_soft_buy,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec

# =========================
# ЭНДПОИНТЫ
# =========================

@app.get("/health", summary="Service availability check")
def health():
    return {"status": "ok"}

@app.get(
    "/assets",
    response_model=List[AssetSummary],
    summary="List of assets and latest T+N forecasts",
)
def list_assets(
    horizon_days: int = DEFAULT_HORIZON,
    db: Session = Depends(get_db),
):
    _validate_horizon(horizon_days)
    result: List[AssetSummary] = []

    for sym in SUPPORTED_ASSETS:
        try:
            rec = _get_last_prediction_from_db(db, sym, horizon_days)
            result.append(
                AssetSummary(
                    symbol=sym,
                    horizon_days=horizon_days,
                    last_up_prob=rec.up_prob,
                    last_verdict=rec.ux_verdict,
                    last_pred_date=rec.pred_date,
                )
            )
        except HTTPException as e:
            print(f"[WARN] asset {sym}, horizon T+{horizon_days}: {e.detail}")
            continue

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"There are no forecasts available for any asset for T+{horizon_days}.",
        )
    return result

@app.get(
    "/prediction/{symbol}",
    response_model=Prediction,
    summary="Latest forecast for the asset for the selected horizon",
)
def get_prediction(
    symbol: str,
    horizon_days: int = DEFAULT_HORIZON,
    db: Session = Depends(get_db),
):
    symbol = symbol.upper()
    if symbol not in SUPPORTED_ASSETS:
        raise HTTPException(status_code=404, detail=f"Asset {symbol} is not supported")

    _validate_horizon(horizon_days)
    rec = _get_last_prediction_from_db(db, symbol, horizon_days)

    return Prediction(
        symbol=rec.symbol,
        horizon_days=rec.horizon_days,
        asof_time=rec.asof_time,
        pred_date=rec.pred_date,
        up_prob=rec.up_prob,
        ux_verdict=rec.ux_verdict,
        ux_soft_buy=rec.ux_soft_buy,
    )

@app.get(
    "/online_prediction/{symbol}",
    response_model=Prediction,
    summary="Online forecast from yfinance for the selected asset and horizon",
)
def get_online_prediction(
    symbol: str,
    horizon_days: int = DEFAULT_HORIZON,
    db: Session = Depends(get_db),
):
    symbol = symbol.upper()
    if symbol not in SUPPORTED_ASSETS:
        raise HTTPException(status_code=404, detail=f"Asset {symbol} is not supported")

    _validate_horizon(horizon_days)

    try:
        res = predict_live(symbol, horizon_days=horizon_days)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Online inference error for {symbol}, T+{horizon_days}: {e}",
        )

    asof_time = pd.to_datetime(res["asof_time"])
    pred_date = pd.to_datetime(res["pred_date"])
    up_prob = float(res["prob_up_calibrated"])
    ux_soft_buy = bool(res["decision_long"])
    ux_verdict = str(res["ux_verdict"])

    # upsert online forecast into DB (for history / signals / charts)
    try:
        _upsert_online_prediction(
            db=db,
            symbol=symbol,
            horizon_days=horizon_days,
            asof_time=asof_time.to_pydatetime(),
            pred_date=pred_date.to_pydatetime(),
            up_prob=up_prob,
            ux_verdict=ux_verdict,
            ux_soft_buy=ux_soft_buy,
        )
    except Exception as e:
        db.rollback()
        print(f"[WARN] cannot persist online prediction for {symbol}, T+{horizon_days}: {e}")

    return Prediction(
        symbol=symbol,
        horizon_days=horizon_days,
        asof_time=asof_time,
        pred_date=pred_date,
        up_prob=up_prob,
        ux_verdict=ux_verdict,
        ux_soft_buy=ux_soft_buy,
    )

@app.get(
    "/prediction/{symbol}/history",
    response_model=List[PredictionHistoryPoint],
    summary="History of forecasts for the asset on the chart",
)
def get_prediction_history(
    symbol: str,
    limit: int = 30,
    horizon_days: int = DEFAULT_HORIZON,
    db: Session = Depends(get_db),
):
    symbol = symbol.upper()
    if symbol not in SUPPORTED_ASSETS:
        raise HTTPException(status_code=404, detail=f"Asset {symbol} is not supported")

    _validate_horizon(horizon_days)

    rows = _get_prediction_history_from_db(db, symbol, horizon_days, limit)

    history: List[PredictionHistoryPoint] = []
    for rec in rows:
        history.append(
            PredictionHistoryPoint(
                symbol=rec.symbol,
                horizon_days=rec.horizon_days,
                asof_time=rec.asof_time,
                pred_date=rec.pred_date,
                up_prob=rec.up_prob,
                ux_verdict=rec.ux_verdict,
            )
        )
    return history

@app.get(
    "/signals",
    response_model=List[SignalCard],
    summary="Advanced signals across all assets",
)
def get_signals(
    horizon_days: int = DEFAULT_HORIZON,
    db: Session = Depends(get_db),
):
    _validate_horizon(horizon_days)
    result: List[SignalCard] = []

    for sym in SUPPORTED_ASSETS:
        try:
            pred = _get_last_prediction_from_db(db, sym, horizon_days)
        except HTTPException as e:
            print(f"[WARN] signal {sym}, horizon T+{horizon_days}: {e.detail}")
            continue

        price, change_24h = _load_features_last(sym)
        stats_rec = _get_model_stats_from_db(db, sym, horizon_days)

        strategy_return = stats_rec.strategy_return if stats_rec else None
        sharpe = stats_rec.sharpe if stats_rec else None

        card = SignalCard(
            symbol=sym,
            horizon_days=horizon_days,
            verdict=pred.ux_verdict,
            up_prob=pred.up_prob,
            confidence=_confidence_from_prob(pred.up_prob),
            price=price,
            change_24h=change_24h,
            strategy_return=strategy_return,
            sharpe=sharpe,
            last_update=pred.pred_date,
        )
        result.append(card)

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No signals for any asset for T+{horizon_days}.",
        )
    return result

@app.get(
    "/stats",
    response_model=List[ModelStats],
    summary="Summary metrics of models by assets",
)
def get_stats(
    horizon_days: int = DEFAULT_HORIZON,
    db: Session = Depends(get_db),
):
    _validate_horizon(horizon_days)

    rows = (
        db.query(ModelStatsRecord)
        .filter(ModelStatsRecord.horizon_days == horizon_days)
        .all()
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No model metrics available for the horizon T+{horizon_days}.",
        )

    stats: List[ModelStats] = []
    for rec in rows:
        strat = rec.strategy_return
        bh = rec.buy_hold_return
        sharpe = rec.sharpe
        win_rate = rec.win_rate
        t_start = rec.test_period_start
        t_end = rec.test_period_end

        # Fallback из meta, if there are no returns in the database
        if strat is None or bh is None:
            fb = _load_business_from_meta(rec.symbol, rec.horizon_days)
            if fb:
                if strat is None and fb.get("strategy_return") is not None:
                    strat = fb["strategy_return"]
                if bh is None and fb.get("buy_hold_return") is not None:
                    bh = fb["buy_hold_return"]
                if sharpe is None and fb.get("sharpe") is not None:
                    sharpe = fb["sharpe"]
                if win_rate is None and fb.get("win_rate") is not None:
                    win_rate = fb["win_rate"]
                if t_start is None and fb.get("test_period_start") is not None:
                    t_start = fb["test_period_start"]
                if t_end is None and fb.get("test_period_end") is not None:
                    t_end = fb["test_period_end"]

        stats.append(
            ModelStats(
                symbol=rec.symbol,
                horizon_days=rec.horizon_days,
                bal_acc=rec.bal_acc,
                auc=rec.auc,
                strategy_return=strat,
                buy_hold_return=bh,
                sharpe=sharpe,
                win_rate=win_rate,
                test_period_start=t_start,
                test_period_end=t_end,
            )
        )

    return stats

@app.get("/", include_in_schema=False)
def root():
    return {"message": "StockAI API is operational. Documentation: /docs"}


