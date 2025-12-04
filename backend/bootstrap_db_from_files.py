# backend/bootstrap_db_from_files.py
from pathlib import Path

from dotenv import load_dotenv

# Загрузить backend/.env (то же место, что и в main.py)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

from pathlib import Path
import json

import pandas as pd
from sqlalchemy.orm import Session

from .database import SessionLocal, engine, Base
from .models_db import PredictionRecord, ModelStatsRecord

# === Config: paths / assets / model versions ===
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

SUPPORTED_ASSETS = ["BTC", "ETH", "TSLA", "AAPL"]

# mapping: horizon_days -> model_version_suffix
HORIZON_MODEL_VERSION = {
    1: "hybrid_xgb_multi_tplus1_classic_gate_v10_1_v11ux",
    3: "hybrid_xgb_multi_tplus3_classic_gate_v10_1_v11ux_v2",
    7: "hybrid_xgb_multi_tplus7_classic_gate_v10_1_v11ux",
}
SUPPORTED_HORIZONS = sorted(HORIZON_MODEL_VERSION.keys())


def load_predictions_for_symbol(db: Session, symbol: str, horizon_days: int):
    """
    Loads predictions_ux_* for a specific asset and horizon T+{horizon_days}.
    Expects files in the format:
        predictions_ux_{symbol}_hybrid_xgb_multi_tplus{horizon_days}_*.csv
    Takes the most recent file (alphabetically last).
    """
    pattern = f"predictions_ux_{symbol}_hybrid_xgb_multi_tplus{horizon_days}_*.csv"
    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        print(f"[WARN] predictions CSV not found for {symbol}, T+{horizon_days} (pattern {pattern})")
        return

    csv_path = files[-1]
    print(f"[INFO] loading predictions for {symbol}, T+{horizon_days} from {csv_path.name}")

    df = pd.read_csv(csv_path)

    required_cols = {
        "symbol",
        "asof_time",
        "pred_date",
        "up_prob",
        "ux_verdict",
        "ux_soft_buy",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] {csv_path}: missing columns {missing}, skip {symbol} T+{horizon_days}")
        return

    if df.empty:
        print(f"[WARN] {csv_path} is empty for {symbol}, T+{horizon_days}")
        return

    df["asof_time"] = pd.to_datetime(df["asof_time"], errors="coerce")
    df["pred_date"] = pd.to_datetime(df["pred_date"], errors="coerce")
    df = df.dropna(subset=["asof_time", "pred_date"])

    # cleaning up old predictions for this symbol+horizon
    deleted = (
        db.query(PredictionRecord)
        .filter(
            PredictionRecord.symbol == symbol,
            PredictionRecord.horizon_days == horizon_days,
        )
        .delete(synchronize_session=False)
    )
    if deleted:
        print(f"[INFO] deleted {deleted} old prediction rows for {symbol}, T+{horizon_days}")

    rows = []
    for _, row in df.iterrows():
        rec = PredictionRecord(
            symbol=symbol,
            horizon_days=horizon_days,
            asof_time=row["asof_time"].to_pydatetime(),
            pred_date=row["pred_date"].to_pydatetime(),
            up_prob=float(row["up_prob"]),
            ux_verdict=str(row["ux_verdict"]),
            ux_soft_buy=bool(row["ux_soft_buy"]),
        )
        rows.append(rec)

    if not rows:
        print(f"[WARN] no valid rows after cleaning for {symbol}, T+{horizon_days}")
        return

    db.add_all(rows)
    db.commit()
    print(f"[INFO] inserted {len(rows)} prediction rows for {symbol}, T+{horizon_days}")


def load_model_stats_for_symbol(db: Session, symbol: str, horizon_days: int):
    """
    Loads holdout / business metrics for symbol + horizon_days
    from meta file:
        {MODELS_DIR}/{symbol}_{model_version}.meta.json
    model_version берём из HORIZON_MODEL_VERSION[horizon_days].
    """
    model_version = HORIZON_MODEL_VERSION.get(horizon_days)
    if model_version is None:
        print(f"[WARN] no model_version for T+{horizon_days}, skip stats for {symbol}")
        return

    meta_path = MODELS_DIR / f"{symbol}_{model_version}.meta.json"
    if not meta_path.exists():
        print(f"[WARN] meta file not found for {symbol}, T+{horizon_days}: {meta_path.name}")
        return

    print(f"[INFO] loading model stats for {symbol}, T+{horizon_days} from {meta_path.name}")

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"[ERROR] cannot read meta for {symbol}, T+{horizon_days}: {e}")
        return

    holdout = meta.get("holdout_metrics_raw", {}) or {}
    biz = meta.get("business", {}) or {}
    test_period = biz.get("test_period", [None, None])

    test_start = None
    test_end = None
    if isinstance(test_period, list) and len(test_period) == 2:
        if test_period[0]:
            test_start = pd.to_datetime(test_period[0]).to_pydatetime()
        if test_period[1]:
            test_end = pd.to_datetime(test_period[1]).to_pydatetime()

    # cleaning up old records for this symbol+horizon
    deleted = (
        db.query(ModelStatsRecord)
        .filter(
            ModelStatsRecord.symbol == symbol,
            ModelStatsRecord.horizon_days == horizon_days,
        )
        .delete(synchronize_session=False)
    )
    if deleted:
        print(f"[INFO] deleted old stats rows for {symbol}, T+{horizon_days}: {deleted}")

    rec = ModelStatsRecord(
        symbol=symbol,
        horizon_days=horizon_days,
        bal_acc=holdout.get("bal_acc"),
        auc=holdout.get("auc"),
        strategy_return=biz.get("Strategy Return (Long-only)"),
        buy_hold_return=biz.get("Buy&Hold Return"),
        sharpe=biz.get("Sharpe (all, long-only)"),
        win_rate=biz.get("Win Rate (executed)"),
        test_period_start=test_start,
        test_period_end=test_end,
    )
    db.add(rec)
    db.commit()
    print(f"[INFO] upserted stats row for {symbol}, T+{horizon_days}")


def main():
    # create tables if they do not already exist
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        for sym in SUPPORTED_ASSETS:
            for h in SUPPORTED_HORIZONS:
                load_predictions_for_symbol(db, sym, h)
                load_model_stats_for_symbol(db, sym, h)
    finally:
        db.close()


if __name__ == "__main__":
    main()
