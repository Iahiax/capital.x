"""AI-assisted live trading strategy."""

from __future__ import annotations

import pandas as pd

from drift_monitor import feature_drift
from features import create_pro_features
from model import (
    add_ai_prob,
    generate_oof_ai_prob,
    train_meta_model,
    train_regime_models,
)
from signals import generate_signals

AI_STATE = {
    "regime_models": None,
    "meta_model": None,
    "features_df": None,
    "candles_df": None,
    "reference_features_df": None,
    "drift_status": {},
    "last_audit": {},
}


def init_ai_model(history_df: pd.DataFrame) -> None:
    """Train the live strategy models from historical candles once."""

    df_feat = create_pro_features(history_df)
    oof_predictions = generate_oof_ai_prob(df_feat)
    regime_models = train_regime_models(df_feat, persist=False)
    df_feat = add_ai_prob(df_feat, regime_models)
    meta_model = train_meta_model(df_feat, oof_predictions, persist=False)

    AI_STATE["regime_models"] = regime_models
    AI_STATE["meta_model"] = meta_model
    AI_STATE["features_df"] = df_feat
    AI_STATE["candles_df"] = history_df.copy()
    AI_STATE["reference_features_df"] = df_feat.copy()
    AI_STATE["drift_status"] = {}


def refresh_ai_features(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild live features after new candles without retraining models."""

    regime_models = AI_STATE["regime_models"]
    if regime_models is None:
        raise RuntimeError("AI strategy must be initialized before refreshing features")

    refreshed = create_pro_features(candles_df)
    refreshed = add_ai_prob(refreshed, regime_models)
    AI_STATE["features_df"] = refreshed
    AI_STATE["candles_df"] = candles_df.copy()

    reference = AI_STATE["reference_features_df"]
    if reference is not None:
        AI_STATE["drift_status"] = feature_drift(reference, refreshed)
    return refreshed


def generate_signal() -> tuple[str | None, dict]:
    """Generate the latest live BUY/SELL signal and its execution metadata."""

    meta_model = AI_STATE["meta_model"]
    df_feat = AI_STATE["features_df"]

    if meta_model is None or df_feat is None or df_feat.empty:
        print("⚠️ نماذج الذكاء الاصطناعي غير مهيأة بعد – لا توجد إشارة")
        return None, {}

    signals_df = generate_signals(
        df_feat,
        news_blackout=None,
        meta_model=meta_model,
    )
    AI_STATE["last_audit"] = signals_df.attrs.get("rejection_counts", {})

    if signals_df.empty:
        return None, {"rejection_counts": AI_STATE["last_audit"]}

    signal_row = signals_df.iloc[-1]
    latest_candle_time = df_feat.index[-1]
    if pd.Timestamp(signal_row["Time"]) != pd.Timestamp(latest_candle_time):
        return None, {"rejection_counts": AI_STATE["last_audit"], "status": "stale_signal"}

    signal_type = "BUY" if signal_row["Type"] == "LONG" else "SELL"
    score = signal_row.get("Score", 0.0)
    atr = signal_row.get("ATR", 0.001)
    regime = signal_row.get("Regime", "UNKNOWN")

    sl = max(atr * (1.0 - min(score / 200.0, 0.5)), 1e-6)
    tp = max(atr * (1.0 + min(score / 150.0, 1.0)), sl)
    stop_pips = max(sl * 10000, 10.0)

    return signal_type, {
        "signal_time": signal_row["Time"],
        "entry_price": float(signal_row["Price"]),
        "regime": regime,
        "stop_pips": stop_pips,
        "pip_value": 0.0001,
        "stop_distance": sl,
        "take_profit_distance": tp,
        "score": score,
        "atr": atr,
    }


def update_ai_context(latest_df: pd.DataFrame) -> None:
    """Compatibility wrapper for refreshing features and drift status."""

    if AI_STATE["meta_model"] is None:
        raise RuntimeError("AI model must be initialized before refreshing context")
    refresh_ai_features(latest_df)


def get_drift_status() -> dict[str, float]:
    return dict(AI_STATE["drift_status"])