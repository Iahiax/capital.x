"""Optional hyperparameter optimization for the signal filters."""

from __future__ import annotations

import logging

import pandas as pd

from backtest import run_backtest
from config import INITIAL_EQUITY, RISK_PER_TRADE
from signals import generate_signals
from walk_forward import probabilistic_sharpe_ratio

logger = logging.getLogger(__name__)

try:
    import optuna
except ImportError:
    optuna = None


def _build_trades(signals_df: pd.DataFrame) -> pd.DataFrame:
    if signals_df.empty:
        return pd.DataFrame(columns=["Time", "Type", "Entry", "SL", "TP", "Size"])
    atr = signals_df["ATR"].clip(lower=1e-6)
    score = signals_df["Score"]
    sl = (atr * (1.0 - (score / 200.0).clip(upper=0.5))).clip(lower=1e-6)
    tp = (atr * (1.0 + (score / 150.0).clip(upper=1.0))).clip(lower=sl)
    quality = (score / 100.0).clip(0.5, 1.5)
    return pd.DataFrame(
        {
            "Time": signals_df["Time"],
            "Type": signals_df["Type"],
            "Entry": signals_df["Price"],
            "SL": sl,
            "TP": tp,
            "Size": INITIAL_EQUITY * RISK_PER_TRADE * quality / sl,
            "Score": score,
        }
    )


def objective(trial, df_feat: pd.DataFrame, df_prices: pd.DataFrame) -> float:
    parameters = {
        "ai_long": trial.suggest_float("ai_long", 0.55, 0.8),
        "ai_short": trial.suggest_float("ai_short", 0.2, 0.45),
        "rvol_min": trial.suggest_float("rvol_min", 0.8, 1.5),
        "shock_max": trial.suggest_float("shock_max", 1.0, 2.0),
        "noise_max": trial.suggest_float("noise_max", 1.2, 2.0),
        "mq_min": trial.suggest_float("mq_min", -0.2, 0.3),
        "score_min": trial.suggest_float("score_min", 50, 80),
    }
    signals_df = generate_signals(df_feat, parameters=parameters)
    stats = run_backtest(_build_trades(signals_df), df_prices)
    executed = stats["trades_df"].query("Status == 'EXECUTED'")
    psr = probabilistic_sharpe_ratio(executed["PnL"])
    drawdown_penalty = stats["max_drawdown"] / max(INITIAL_EQUITY, 1.0)
    return float(psr * 100 + stats["profit"] / max(INITIAL_EQUITY, 1.0) - drawdown_penalty)


def run_optuna(
    df_feat: pd.DataFrame, df_prices: pd.DataFrame, n_trials: int = 30
) -> dict:
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if optuna is None:
        logger.warning("Optuna is not installed; using safe default filter parameters.")
        return {
            "ai_long": 0.55,
            "ai_short": 0.45,
            "rvol_min": 1.2,
            "shock_max": 1.5,
            "noise_max": 1.5,
            "mq_min": 0.0,
            "score_min": 60.0,
        }

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, df_feat, df_prices), n_trials=n_trials
    )
    logger.info("Best params: %s", study.best_params)
    logger.info("Best score: %s", study.best_value)
    return study.best_params