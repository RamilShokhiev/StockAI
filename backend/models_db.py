# backend/models_db.py
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    UniqueConstraint,
)
from .database import Base


class PredictionRecord(Base):
    """
    Table `predictions` — offline/online forecasts T+N.
    Currently we use horizon_days=1 (T+1), later we will add 3/7.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    symbol = Column(String(10), index=True, nullable=False)
    horizon_days = Column(Integer, index=True, nullable=False, default=1)

    asof_time = Column(DateTime, nullable=False)   # when the model generated the forecast
    pred_date = Column(DateTime, nullable=False)   # which date the forecast is for

    up_prob = Column(Float, nullable=False)
    ux_verdict = Column(String(50), nullable=False)
    ux_soft_buy = Column(Boolean, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "horizon_days",
            "asof_time",
            name="uq_symbol_horizon_asof",
        ),
    )


class ModelStatsRecord(Base):
    """
    Table `model_stats` — aggregated backtest metrics for each asset and horizon.
    Logically we take the data from *.meta.json (holdout_metrics_raw + business).
    """
    __tablename__ = "model_stats"

    id = Column(Integer, primary_key=True, index=True)

    symbol = Column(String(10), index=True, nullable=False)
    horizon_days = Column(Integer, index=True, nullable=False, default=1)

    bal_acc = Column(Float, nullable=True)
    auc = Column(Float, nullable=True)
    strategy_return = Column(Float, nullable=True)
    buy_hold_return = Column(Float, nullable=True)
    sharpe = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)

    test_period_start = Column(DateTime, nullable=True)
    test_period_end = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "horizon_days",
            name="uq_symbol_horizon_stats",
        ),
    )
