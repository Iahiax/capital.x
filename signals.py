"""Vectorized signal generation with directional and meta-model filters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meta_model import MetaDecisionModel


def _score_frame(df: pd.DataFrame, short: pd.Series) -> pd.Series:
    directional_ai = np.where(short, 1 - df["AI_Prob"], df["AI_Prob"])
    directional_trend = np.where(short, -df["TrendStrength"], df["TrendStrength"])
    directional_pressure = np.where(short, df["SellPressure"], df["BuyPressure"])
    directional_aggression = np.where(
        short, df["AggressiveSell"], df["AggressiveBuy"]
    )
    opposite_aggression = np.where(
        short, df["AggressiveBuy"], df["AggressiveSell"]
    )
    return (
        directional_ai * 50
        + (directional_trend / (df["ATR"] + 1e-6)) * 10
        + directional_pressure * 10
        + df["RVOL"] * 10
        - df["ShockIndex"] * 10
        - df["NoiseIndex"] * 10
        + directional_aggression * 5
        - opposite_aggression * 5
    )


def compute_signal_score(row, direction: str | None = None) -> float:
    if direction is None:
        direction = "SHORT" if row.get("Regime") == -1 else "LONG"
    values = pd.DataFrame([row])
    return float(_score_frame(values, pd.Series([direction == "SHORT"])).iloc[0])


def dynamic_thresholds(df):
    if df.empty:
        return 0.55, 0.45
    vol = df["ATR"].rolling(200).mean().iloc[-1]
    # Calibrated probabilities need lower thresholds than raw XGB scores.
    base_ai_long = 0.55
    base_ai_short = 0.45
    avg_vol = df["ATR"].mean()
    if pd.isna(vol):
        vol = avg_vol
    if vol > avg_vol:
        base_ai_long += 0.05
        base_ai_short -= 0.05
    else:
        base_ai_long -= 0.05
        base_ai_short += 0.05
    return base_ai_long, base_ai_short


def generate_signals(
    df_feat: pd.DataFrame,
    news_blackout=None,
    meta_model: MetaDecisionModel | None = None,
    parameters: dict | None = None,
    enabled_filters: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Generate signals with individually switchable research filters.

    The default keeps every production filter enabled. ``enabled_filters`` is
    intended for research ablations only; callers can remove one component
    without changing thresholds, scores, sizing inputs, or the cost model.
    """
    columns = ["Time", "Type", "Price", "ATR", "Score", "MetaProb", "Regime"]
    if df_feat.empty:
        return pd.DataFrame(columns=columns)

    parameters = parameters or {}
    filters = {
        "news": True,
        "meta": True,
        "session": True,
        "regime": True,
        "market_quality": True,
    }
    if enabled_filters:
        unknown = set(enabled_filters).difference(filters)
        if unknown:
            raise ValueError(f"Unknown signal filters: {sorted(unknown)}")
        filters.update(enabled_filters)

    ai_long_thr, ai_short_thr = dynamic_thresholds(df_feat)
    ai_long_thr = parameters.get("ai_long", ai_long_thr)
    ai_short_thr = parameters.get("ai_short", ai_short_thr)
    rvol_min = parameters.get("rvol_min", 1.2)
    shock_max = parameters.get("shock_max", 1.5)
    noise_max = parameters.get("noise_max", 1.5)
    mq_min = parameters.get("mq_min", 0.0)
    score_min = parameters.get("score_min", 60.0)

    session = df_feat["Hour"].between(8, 11) | df_feat["Hour"].between(14, 17)
    blackout = (
        pd.Series(False, index=df_feat.index)
        if news_blackout is None
        else news_blackout.reindex(df_feat.index, fill_value=False).astype(bool)
    )
    session_filter = (
        session if filters["session"] else pd.Series(True, index=df_feat.index)
    )
    news_filter = (
        ~blackout if filters["news"] else pd.Series(True, index=df_feat.index)
    )
    regime_filter = (
        df_feat["Regime"].ne(2)
        if filters["regime"]
        else pd.Series(True, index=df_feat.index)
    )
    quality_filter = (
        df_feat["MarketQuality"].ge(mq_min)
        if filters["market_quality"]
        else pd.Series(True, index=df_feat.index)
    )
    common = (
        session_filter
        & news_filter
        & regime_filter
        & quality_filter
    )
    short_score = _score_frame(df_feat, pd.Series(True, index=df_feat.index))
    long_score = _score_frame(df_feat, pd.Series(False, index=df_feat.index))

    regime_long = (
        df_feat["Regime"].eq(1)
        if filters["regime"]
        else pd.Series(True, index=df_feat.index)
    )
    regime_short = (
        df_feat["Regime"].eq(-1)
        if filters["regime"]
        else pd.Series(True, index=df_feat.index)
    )
    long_mask = common & regime_long
    long_mask &= (
        long_score.ge(score_min)
        & df_feat["AI_Prob"].gt(ai_long_thr)
        & df_feat["TrendStrength"].gt(0)
        & df_feat["Kalman_Fast_Slope"].gt(0)
        & df_feat["BuyPressure"].gt(0.6)
        & df_feat["RVOL"].gt(rvol_min)
        & df_feat["ShockIndex"].lt(shock_max)
        & df_feat["SmartDiv"].gt(0)
        & df_feat["NoiseIndex"].lt(noise_max)
    )

    short_mask = common & regime_short
    short_mask &= (
        short_score.ge(score_min)
        & df_feat["AI_Prob"].lt(ai_short_thr)
        & df_feat["TrendStrength"].lt(0)
        & df_feat["Kalman_Fast_Slope"].lt(0)
        & df_feat["SellPressure"].gt(0.6)
        & df_feat["RVOL"].gt(rvol_min)
        & df_feat["ShockIndex"].lt(shock_max)
        & df_feat["SmartDiv"].lt(0)
        & df_feat["NoiseIndex"].lt(noise_max)
    )

    candidate = long_mask | short_mask
    candidate_score = pd.Series(
        np.where(long_mask, long_score, short_score), index=df_feat.index
    )
    meta_prob = pd.Series(1.0, index=df_feat.index, dtype=float)
    if meta_model is not None and candidate.any():
        meta_prob.loc[candidate] = meta_model.predict_prob_frame(
            df_feat.loc[candidate, "AI_Prob"].to_numpy(),
            candidate_score.loc[candidate].to_numpy(),
            df_feat.loc[candidate, "Regime"].to_numpy(),
            df_feat.loc[candidate, "MarketQuality"].to_numpy(),
        )

    model_agreement = df_feat.get(
        "AI_Disagreement", pd.Series(0.0, index=df_feat.index)
    )
    if filters["meta"]:
        accepted = candidate & meta_prob.ge(0.65) & model_agreement.le(0.35)
    else:
        accepted = candidate
    if not accepted.any():
        empty = pd.DataFrame(columns=columns)
        empty.attrs["rejection_counts"] = {
            "candidates": int(candidate.sum()),
            "meta_rejected": int((candidate & meta_prob.lt(0.65)).sum()),
            "model_disagreement": int(
                (candidate & model_agreement.gt(0.35)).sum()
            ),
        }
        return empty

    result = pd.DataFrame(
        {
            "Time": df_feat.index[accepted],
            "Type": np.where(long_mask.loc[accepted], "LONG", "SHORT"),
            "Price": df_feat.loc[accepted, "Close"].to_numpy(),
            "ATR": df_feat.loc[accepted, "ATR"].to_numpy(),
            "Score": np.where(
                long_mask.loc[accepted],
                long_score.loc[accepted],
                short_score.loc[accepted],
            ),
            "MetaProb": meta_prob.loc[accepted].to_numpy(),
            "Regime": df_feat.loc[accepted, "Regime"].to_numpy(),
        }
    )
    result = result[columns].reset_index(drop=True)
    result.attrs["rejection_counts"] = {
        "candidates": int(candidate.sum()),
        "accepted": int(accepted.sum()),
        "meta_rejected": int((candidate & meta_prob.lt(0.65)).sum()),
        "model_disagreement": int((candidate & model_agreement.gt(0.35)).sum()),
    }
    return result