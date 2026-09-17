"""Leakage-aware walk-forward validation for the research pipeline."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from backtest import run_backtest
from config import INITIAL_EQUITY, MAX_LOOKAHEAD_MINUTES, RISK_PER_TRADE
from features import create_pro_features
from model import (
    DEFAULT_TARGET_COLUMN,
    SUPPORTED_TARGET_COLUMNS,
    add_ai_prob,
    generate_oof_ai_prob,
    train_meta_model,
    train_regime_models,
)
from signals import generate_signals

logger = logging.getLogger(__name__)

ABLATION_FILTERS = {
    "full_strategy": {},
    "without_news_filter": {"news": False},
    "without_meta_model": {"meta": False},
    "without_session_filter": {"session": False},
    "without_regime_filter": {"regime": False},
    "without_market_quality_filter": {"market_quality": False},
}

ABLATION_DESCRIPTIONS = {
    "full_strategy": "All production filters enabled",
    "without_news_filter": "News blackout disabled",
    "without_meta_model": "Meta-Model probability and disagreement gates disabled",
    "without_session_filter": "Trading-session hours gate disabled",
    "without_regime_filter": "Regime direction and chaos gates disabled",
    "without_market_quality_filter": "Market-quality threshold gate disabled",
}


def _empty_regime_metrics() -> dict[str, dict]:
    return {
        regime: {
            "profit": 0.0,
            "pnl": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": None,
            "num_trades": 0,
            "trades": 0,
        }
        for regime in ("uptrend", "downtrend", "range", "chaos")
    }


def probabilistic_sharpe_ratio(pnls, benchmark: float = 0.0) -> float:
    """Estimate the probability that the observed Sharpe exceeds benchmark."""

    values = np.asarray(list(pnls), dtype=float)
    if values.size < 2 or np.std(values, ddof=1) == 0:
        return 0.0
    sharpe = (values.mean() - benchmark) / values.std(ddof=1)
    skew = pd.Series(values).skew()
    kurtosis = pd.Series(values).kurtosis() + 3
    denominator = math.sqrt(
        max(1e-12, 1 - skew * sharpe + ((kurtosis - 1) / 4) * sharpe**2)
    )
    z_score = sharpe * math.sqrt(values.size - 1) / denominator
    return float(0.5 * (1 + math.erf(z_score / math.sqrt(2))))


def _signals_to_trades(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=["Time", "Type", "Entry", "SL", "TP", "Size"])
    atr = signals["ATR"].clip(lower=1e-6)
    score = signals["Score"]
    sl = atr * (1.0 - (score / 200.0).clip(upper=0.5))
    tp = atr * (1.0 + (score / 150.0).clip(upper=1.0))
    risk_amount = INITIAL_EQUITY * RISK_PER_TRADE * (score / 100).clip(0.5, 1.5)
    return pd.DataFrame(
        {
            "Time": signals["Time"],
            "Type": signals["Type"],
            "Entry": signals["Price"],
            "SL": sl,
            "TP": tp,
            "Size": risk_amount / sl,
            "Score": score,
            "Regime": signals["Regime"].to_numpy(),
        }
    )


def walk_forward(
    df,
    train_window="90D",
    test_window="30D",
    trainer_fn=None,
    eval_fn=None,
    embargo="0min",
    purge_window=None,
):
    """Run rolling chronological folds without training on purged labels.

    ``df_train`` ends at ``train_end - embargo`` while ``df_test`` starts at
    ``train_end``. This keeps future-label observations that could overlap the
    first test observations out of the training set. The default remains zero
    for the small generic helper; strategy validation supplies a label horizon.
    """

    if trainer_fn is None or eval_fn is None:
        raise ValueError("trainer_fn and eval_fn are required")
    if df.empty:
        return []
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("walk_forward requires a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("walk_forward requires a sorted DatetimeIndex")
    if df.index.has_duplicates:
        raise ValueError("walk_forward requires unique timestamps")

    results = []
    start = df.index.min()
    end = df.index.max()
    current = start
    train_delta = pd.Timedelta(train_window)
    test_delta = pd.Timedelta(test_window)
    purge_delta = pd.Timedelta(
        embargo if purge_window is None else purge_window
    )
    if train_delta <= pd.Timedelta(0) or test_delta <= pd.Timedelta(0):
        raise ValueError("train_window and test_window must be positive")
    if purge_delta < pd.Timedelta(0):
        raise ValueError("embargo/purge_window must be non-negative")

    while current + train_delta + test_delta <= end:
        train_end = current + train_delta
        test_end = train_end + test_delta
        purged_train_end = train_end - purge_delta
        df_train = df[(df.index >= current) & (df.index < purged_train_end)]
        df_test = df[(df.index >= train_end) & (df.index < test_end)]
        if df_train.empty or df_test.empty:
            current = train_end
            continue
        model = trainer_fn(df_train)
        result = eval_fn(model, df_test)
        if isinstance(result, dict):
            result = {
                **result,
                "train_start": df_train.index.min(),
                "train_end": df_train.index.max(),
                "test_start": df_test.index.min(),
                "test_end": df_test.index.max(),
                "purged_until": purged_train_end,
            }
        results.append(result)
        current = train_end
    return results


def validate_strategy_walk_forward(
    df: pd.DataFrame,
    train_window="90D",
    test_window="30D",
    min_successive_windows: int = 3,
    label_horizon="3min",
) -> dict:
    """Validate only on forward data and return an auditable OOS report.

    Features are built once from the complete candle history because all model
    inputs are causal rolling values. Models are still fitted separately on
    each train fold. Signals from all forward folds are replayed together on
    the original price stream so overlap and transaction costs are counted
    exactly once.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("validate_strategy_walk_forward requires a DatetimeIndex")
    prices = df.sort_index()
    features = create_pro_features(prices)
    if features.empty:
        return {
            "approved": False,
            "windows": [],
            "aggregate": _empty_validation_metrics(),
            "aggregate_passed": False,
            "regime_metrics": _empty_regime_metrics(),
            "costs": _cost_configuration(),
            "consecutive_passes": 0,
            "min_successive_windows": min_successive_windows,
        }
    oos_trades: list[pd.DataFrame] = []

    def trainer_fn(train_features):
        oof = generate_oof_ai_prob(train_features)
        models = train_regime_models(train_features, persist=False)
        enriched = add_ai_prob(train_features, models)
        meta = train_meta_model(enriched, oof, persist=False)
        return models, meta

    def eval_fn(model_bundle, test_features):
        models, meta = model_bundle
        test_features = add_ai_prob(test_features, models)
        signals = generate_signals(test_features, meta_model=meta)
        trades = _signals_to_trades(signals)
        if not trades.empty:
            oos_trades.append(trades)
        # Fold metrics are computed on the fold's unseen interval plus enough
        # bars to resolve exits that occur just after the reporting boundary.
        test_start = test_features.index.min()
        test_end = test_features.index.max()
        price_end = test_end + pd.Timedelta(MAX_LOOKAHEAD_MINUTES, unit="min")
        test_prices = prices.loc[
            (prices.index >= test_start) & (prices.index <= price_end)
        ]
        stats = run_backtest(trades, test_prices)
        executed = stats["trades_df"].query("Status == 'EXECUTED'")
        pnls = executed["PnL"].to_numpy()
        return {
            **_metrics_from_stats(stats, pnls),
            "win_rate": float(stats["win_rate"]),
            "probabilistic_sharpe": probabilistic_sharpe_ratio(pnls),
            "passed": bool(
                len(pnls) >= 2
                and stats["profit"] > 0
                and probabilistic_sharpe_ratio(pnls) >= 0.95
            ),
        }

    results = walk_forward(
        features,
        train_window=train_window,
        test_window=test_window,
        trainer_fn=trainer_fn,
        eval_fn=eval_fn,
        embargo=label_horizon,
    )
    successive = 0
    max_successive = 0
    for result in results:
        successive = successive + 1 if result["passed"] else 0
        max_successive = max(max_successive, successive)

    combined_trades = (
        pd.concat(oos_trades, ignore_index=True)
        if oos_trades
        else _signals_to_trades(pd.DataFrame())
    )
    overall = run_backtest(combined_trades, prices)
    executed = overall["trades_df"].query("Status == 'EXECUTED'")
    aggregate = _metrics_from_stats(overall, executed["PnL"].to_numpy())
    aggregate_passed = bool(
        aggregate["num_trades"] >= 2
        and aggregate["profit"] > 0
        and aggregate["profit_factor"] is not None
        and aggregate["profit_factor"] >= 1.0
    )
    approved = bool(
        max_successive >= min_successive_windows and aggregate_passed
    )
    logger.info(
        "Walk-forward OOS windows=%d, trades=%d, profit=%.2f, approved=%s.",
        len(results),
        aggregate["num_trades"],
        aggregate["profit"],
        approved,
    )
    return {
        "approved": approved,
        "windows": results,
        "aggregate": aggregate,
        "aggregate_passed": aggregate_passed,
        "regime_metrics": overall["regime_metrics"],
        "costs": overall["costs"],
        "trades_df": overall["trades_df"],
        "label_horizon": str(pd.Timedelta(label_horizon)),
        "consecutive_passes": max_successive,
        "min_successive_windows": min_successive_windows,
    }


def _brier_score(targets, probabilities) -> float | None:
    targets = np.asarray(targets, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(targets) == 0:
        return None
    return float(brier_score_loss(targets.astype(int), probabilities))


def _horizon_label(target_column: str) -> str:
    return f"{SUPPORTED_TARGET_COLUMNS[target_column]}m"


def _validate_single_horizon(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    *,
    target_column: str,
    train_window,
    test_window,
    min_successive_windows: int,
) -> dict:
    """Run the existing OOS/paper gate for one target without persistence."""

    if target_column not in SUPPORTED_TARGET_COLUMNS:
        raise ValueError(
            f"Unsupported target column {target_column!r}; "
            f"choose one of {sorted(SUPPORTED_TARGET_COLUMNS)}."
        )

    label_minutes = SUPPORTED_TARGET_COLUMNS[target_column]
    label_horizon = f"{label_minutes}min"
    oos_trades: list[pd.DataFrame] = []
    brier_targets = []
    brier_probabilities = []

    def trainer_fn(train_features):
        oof = generate_oof_ai_prob(
            train_features,
            embargo=label_minutes,
            target_column=target_column,
        )
        models = train_regime_models(
            train_features,
            persist=False,
            target_column=target_column,
        )
        enriched = add_ai_prob(train_features, models)
        meta = train_meta_model(
            enriched,
            oof,
            persist=False,
            target_column=target_column,
        )
        return models, meta

    def eval_fn(model_bundle, test_features):
        models, meta = model_bundle
        test_features = add_ai_prob(test_features, models)
        scored = test_features[target_column].notna()
        if scored.any():
            brier_targets.extend(test_features.loc[scored, target_column].tolist())
            brier_probabilities.extend(test_features.loc[scored, "AI_Prob"].tolist())

        signals = generate_signals(test_features, meta_model=meta)
        trades = _signals_to_trades(signals)
        if not trades.empty:
            oos_trades.append(trades)

        test_start = test_features.index.min()
        test_end = test_features.index.max()
        price_end = test_end + pd.Timedelta(MAX_LOOKAHEAD_MINUTES, unit="min")
        test_prices = prices.loc[
            (prices.index >= test_start) & (prices.index <= price_end)
        ]
        stats = run_backtest(trades, test_prices)
        executed = stats["trades_df"].query("Status == 'EXECUTED'")
        pnls = executed["PnL"].to_numpy()
        psr = probabilistic_sharpe_ratio(pnls)
        return {
            **_metrics_from_stats(stats, pnls),
            "brier_score": _brier_score(
                test_features.loc[scored, target_column],
                test_features.loc[scored, "AI_Prob"],
            ),
            "cost_adjusted_pnl": float(stats["profit"]),
            "win_rate": float(stats["win_rate"]),
            "probabilistic_sharpe": psr,
            "passed": bool(
                len(pnls) >= 2
                and stats["profit"] > 0
                and psr >= 0.95
            ),
        }

    windows = walk_forward(
        features,
        train_window=train_window,
        test_window=test_window,
        trainer_fn=trainer_fn,
        eval_fn=eval_fn,
        embargo=label_horizon,
    )
    successive = 0
    max_successive = 0
    for result in windows:
        successive = successive + 1 if result["passed"] else 0
        max_successive = max(max_successive, successive)

    combined_trades = (
        pd.concat(oos_trades, ignore_index=True)
        if oos_trades
        else _empty_trade_frame()
    )
    overall = run_backtest(combined_trades, prices)
    executed = overall["trades_df"].query("Status == 'EXECUTED'")
    aggregate_pnls = executed["PnL"].to_numpy()
    aggregate_brier = _brier_score(brier_targets, brier_probabilities)
    aggregate_psr = probabilistic_sharpe_ratio(aggregate_pnls)
    aggregate = {
        **_metrics_from_stats(overall, aggregate_pnls),
        "brier_score": aggregate_brier,
        "cost_adjusted_pnl": float(overall["profit"]),
        "probabilistic_sharpe": aggregate_psr,
        "win_rate": float(overall["win_rate"]),
    }
    aggregate_passed = bool(
        aggregate["num_trades"] >= 2
        and aggregate["profit"] > 0
        and aggregate["profit_factor"] is not None
        and aggregate["profit_factor"] >= 1.0
    )
    paper_trading_passed = bool(
        max_successive >= min_successive_windows and aggregate_passed
    )
    return {
        "horizon": _horizon_label(target_column),
        "target_column": target_column,
        "label_horizon": label_horizon,
        "approved": paper_trading_passed,
        "persistence_gate": {
            "passed": paper_trading_passed,
            "requires_non_default_review": target_column != DEFAULT_TARGET_COLUMN,
        },
        "paper_trading_gate": {
            "passed": paper_trading_passed,
            "aggregate_passed": aggregate_passed,
            "consecutive_passes": max_successive,
            "min_successive_windows": min_successive_windows,
        },
        "windows": windows,
        "window_count": len(windows),
        "aggregate": aggregate,
        "aggregate_passed": aggregate_passed,
        "regime_metrics": overall["regime_metrics"],
        "costs": overall["costs"],
        "consecutive_passes": max_successive,
        "min_successive_windows": min_successive_windows,
        # This is deliberately false for non-default horizons. A passing
        # research gate is evidence for review, not an automatic live swap.
        "live_decision_allowed": target_column == DEFAULT_TARGET_COLUMN,
    }


def _write_multi_horizon_report(report: dict, report_dir) -> dict:
    target = Path(report_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "multi_horizon_report.json"
    csv_path = target / "multi_horizon_report.csv"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    rows = []
    for horizon, result in report["horizons"].items():
        aggregate = result["aggregate"]
        rows.append(
            {
                "horizon": horizon,
                "target_column": result["target_column"],
                "brier_score": aggregate["brier_score"],
                "probabilistic_sharpe": aggregate["probabilistic_sharpe"],
                "max_drawdown": aggregate["max_drawdown"],
                "cost_adjusted_pnl": aggregate["cost_adjusted_pnl"],
                "trade_count": aggregate["num_trades"],
                "paper_trading_gate_passed": result["paper_trading_gate"]["passed"],
                "live_decision_allowed": result["live_decision_allowed"],
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"json": str(json_path), "csv": str(csv_path)}


def validate_multi_horizon_forecasts(
    df: pd.DataFrame,
    *,
    train_window="90D",
    test_window="30D",
    min_successive_windows: int = 3,
    horizons=(3, 15, 60),
    report_dir=None,
) -> dict:
    """Compare target horizons using identical chronological OOS folds.

    The function is research-only: every horizon is trained with ``persist=False``
    and the live strategy target remains ``Target_3m``. Each candidate is
    evaluated with the same calibrated OOF process, purge duration, paper replay,
    and approval gate.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("validate_multi_horizon_forecasts requires a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("validate_multi_horizon_forecasts requires sorted candles")
    if df.index.has_duplicates:
        raise ValueError("validate_multi_horizon_forecasts requires unique timestamps")
    if not horizons:
        raise ValueError("At least one horizon is required")

    target_columns = []
    for horizon in horizons:
        if isinstance(horizon, str) and horizon.startswith("Target_"):
            target_column = horizon
        else:
            minutes = int(horizon)
            target_column = f"Target_{minutes}m"
        if target_column not in SUPPORTED_TARGET_COLUMNS:
            raise ValueError(
                f"Unsupported horizon {horizon!r}; "
                f"choose from {sorted(SUPPORTED_TARGET_COLUMNS.values())} minutes."
            )
        if target_column not in target_columns:
            target_columns.append(target_column)

    prices = df.sort_index()
    features = create_pro_features(prices)
    results = {}
    for target_column in target_columns:
        results[_horizon_label(target_column)] = _validate_single_horizon(
            prices,
            features,
            target_column=target_column,
            train_window=train_window,
            test_window=test_window,
            min_successive_windows=min_successive_windows,
        )

    report = {
        "report_version": 1,
        "train_window": str(train_window),
        "test_window": str(test_window),
        "horizons": results,
        "cost_configuration": _cost_configuration(),
        "live_decision_target": DEFAULT_TARGET_COLUMN,
        "research_only": True,
    }
    if report_dir is not None:
        report["report_files"] = _write_multi_horizon_report(report, report_dir)
    return report


# Short alias for callers that use the report terminology.
run_multi_horizon_report = validate_multi_horizon_forecasts


def _ablation_metrics(stats: dict) -> dict:
    """Return the stable, cost-aware columns shared by JSON and CSV reports."""

    return {
        "pnl": float(stats["profit"]),
        "max_drawdown": float(stats["max_drawdown"]),
        "win_rate": float(stats["win_rate"]),
        "profit_factor": (
            float(stats["profit_factor"])
            if stats["profit_factor"] is not None
            else None
        ),
        "trade_count": int(stats["num_trades"]),
        "costs": {
            name: float(value) for name, value in stats["costs"].items()
        },
    }


def _empty_trade_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["Time", "Type", "Entry", "SL", "TP", "Size"])


def _write_ablation_report(report: dict, report_dir) -> dict:
    """Write deterministic machine-readable reports next to walk-forward output."""

    target = Path(report_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "ablation_report.json"
    csv_path = target / "ablation_report.csv"

    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    rows = []
    for name, summary in report["ablations"].items():
        rows.append(
            {
                "ablation": name,
                "description": summary["description"],
                "pnl": summary["pnl"],
                "max_drawdown": summary["max_drawdown"],
                "win_rate": summary["win_rate"],
                "profit_factor": summary["profit_factor"],
                "trade_count": summary["trade_count"],
                "total_cost": summary["costs"]["total"],
                "window_count": len(summary["windows"]),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"json": str(json_path), "csv": str(csv_path)}


def run_ablation_report(
    df: pd.DataFrame,
    *,
    news_blackout=None,
    parameters: dict | None = None,
    train_window="90D",
    test_window="30D",
    label_horizon="3min",
    report_dir=None,
) -> dict:
    """Compare strategy components on identical chronological OOS windows.

    Each fold trains the base and meta models once, then evaluates the full
    strategy and every single-filter removal against the same test candles.
    Trades from each variant are replayed together over all test windows so
    overlap, spread, commission, slippage, and swap are handled by the same
    backtest implementation.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("run_ablation_report requires a DatetimeIndex")
    prices = df.sort_index()
    features = create_pro_features(prices)
    if features.empty:
        report = {
            "report_version": 1,
            "train_window": str(train_window),
            "test_window": str(test_window),
            "label_horizon": str(pd.Timedelta(label_horizon)),
            "cost_configuration": _cost_configuration(),
            "ablations": {
                name: {
                    "description": ABLATION_DESCRIPTIONS[name],
                    **_ablation_metrics(
                        run_backtest(_empty_trade_frame(), prices)
                    ),
                    "windows": [],
                }
                for name in ABLATION_FILTERS
            },
        }
        if report_dir is not None:
            report["report_files"] = _write_ablation_report(report, report_dir)
        return report

    oos_trades = {name: [] for name in ABLATION_FILTERS}

    def trainer_fn(train_features):
        oof = generate_oof_ai_prob(train_features)
        models = train_regime_models(train_features, persist=False)
        enriched = add_ai_prob(train_features, models)
        meta = train_meta_model(enriched, oof, persist=False)
        return models, meta

    def eval_fn(model_bundle, test_features):
        models, meta = model_bundle
        test_features = add_ai_prob(test_features, models)
        test_start = test_features.index.min()
        test_end = test_features.index.max()
        price_end = test_end + pd.Timedelta(MAX_LOOKAHEAD_MINUTES, unit="min")
        test_prices = prices.loc[
            (prices.index >= test_start) & (prices.index <= price_end)
        ]
        fold_results = {}
        for name, filter_overrides in ABLATION_FILTERS.items():
            signals = generate_signals(
                test_features,
                news_blackout=news_blackout,
                meta_model=meta if filter_overrides.get("meta", True) else None,
                parameters=parameters,
                enabled_filters=filter_overrides,
            )
            trades = _signals_to_trades(signals)
            if not trades.empty:
                oos_trades[name].append(trades)
            fold_results[name] = _ablation_metrics(run_backtest(trades, test_prices))
        return {"ablations": fold_results}

    windows = walk_forward(
        features,
        train_window=train_window,
        test_window=test_window,
        trainer_fn=trainer_fn,
        eval_fn=eval_fn,
        embargo=label_horizon,
    )

    ablations = {}
    for name in ABLATION_FILTERS:
        combined_trades = (
            pd.concat(oos_trades[name], ignore_index=True)
            if oos_trades[name]
            else _empty_trade_frame()
        )
        stats = run_backtest(combined_trades, prices)
        summary = {
            "description": ABLATION_DESCRIPTIONS[name],
            **_ablation_metrics(stats),
            "windows": [
                {
                    "train_start": str(window["train_start"]),
                    "train_end": str(window["train_end"]),
                    "test_start": str(window["test_start"]),
                    "test_end": str(window["test_end"]),
                    **window["ablations"][name],
                }
                for window in windows
            ],
        }
        ablations[name] = summary

    report = {
        "report_version": 1,
        "train_window": str(train_window),
        "test_window": str(test_window),
        "label_horizon": str(pd.Timedelta(label_horizon)),
        "window_count": len(windows),
        "cost_configuration": _cost_configuration(),
        "ablations": ablations,
    }
    if report_dir is not None:
        report["report_files"] = _write_ablation_report(report, report_dir)
    return report


def _metrics_from_stats(stats: dict, pnls) -> dict:
    return {
        "profit": float(stats["profit"]),
        "pnl": float(stats["profit"]),
        "max_drawdown": float(stats["max_drawdown"]),
        "num_trades": int(stats["num_trades"]),
        "trades": int(stats["num_trades"]),
        "profit_factor": stats["profit_factor"],
        "costs": stats["costs"],
    }


def _empty_validation_metrics() -> dict:
    return {
        "profit": 0.0,
        "pnl": 0.0,
        "max_drawdown": 0.0,
        "num_trades": 0,
        "trades": 0,
        "profit_factor": None,
    }


def _cost_configuration() -> dict:
    from config import (
        COMMISSION_PER_TRADE,
        SLIPPAGE_ATR_MULTIPLIER,
        SPREAD,
        SWAP_PER_DAY,
    )

    return {
        "spread": SPREAD,
        "commission_per_trade": COMMISSION_PER_TRADE,
        "slippage_atr_multiplier": SLIPPAGE_ATR_MULTIPLIER,
        "swap_per_day": SWAP_PER_DAY,
    }