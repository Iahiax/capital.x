import datetime
import math

import pandas as pd

import trade_engine
from backtest import run_backtest
from capital_allocation import allocate_capital
from data_loader import generate_sample_data
from data_quality import sanitize_candles
from drift_monitor import adversarial_validation, population_stability_index
from features import create_pro_features
from governance import approve_trade
from monte_carlo import run_monte_carlo
from scenario_engine import simulate_market_outage
from scheduler import is_market_closed, should_stop_new_trades
from signals import generate_signals
from trade_store import TradeStore
from walk_forward import (
    probabilistic_sharpe_ratio,
    run_ablation_report,
    walk_forward,
)


def test_sample_data_is_deterministic_and_feature_complete():
    candles_a = generate_sample_data(1_000)
    candles_b = generate_sample_data(1_000)
    pd.testing.assert_frame_equal(candles_a, candles_b)

    features = create_pro_features(candles_a)
    assert len(features) > 0
    assert {
        "ATR",
        "Regime",
        "MarketQuality",
        "Target_3m",
        "Target_15m",
        "Target_60m",
        "Target_3m_cost",
    }.issubset(
        features.columns
    )
    assert features["Target_3m"].isna().sum() == 3


def test_signal_generation_returns_stable_schema():
    features = create_pro_features(generate_sample_data(1_000))
    features["AI_Prob"] = 0.5
    signals = generate_signals(features)
    assert list(signals.columns) == [
        "Time",
        "Type",
        "Price",
        "ATR",
        "Score",
        "MetaProb",
        "Regime",
    ]


def test_signal_filter_switches_preserve_default_behavior():
    features = create_pro_features(generate_sample_data(1_000))
    features["AI_Prob"] = 0.5
    default = generate_signals(features)
    explicit = generate_signals(
        features,
        enabled_filters={
            "news": True,
            "meta": True,
            "session": True,
            "regime": True,
            "market_quality": True,
        },
    )
    pd.testing.assert_frame_equal(default, explicit)


def test_ablation_report_has_shared_windows_and_required_metrics(tmp_path):
    report = run_ablation_report(
        generate_sample_data(2_000),
        train_window="8h",
        test_window="4h",
        report_dir=tmp_path,
    )

    assert report["window_count"] > 0
    assert {
        "full_strategy",
        "without_news_filter",
        "without_meta_model",
        "without_session_filter",
        "without_regime_filter",
        "without_market_quality_filter",
    } == set(report["ablations"])
    for summary in report["ablations"].values():
        assert {
            "pnl",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "trade_count",
            "costs",
            "windows",
        }.issubset(summary)
        assert len(summary["windows"]) == report["window_count"]
    assert (tmp_path / "ablation_report.json").exists()
    assert (tmp_path / "ablation_report.csv").exists()


def test_backtest_prefers_stop_when_both_levels_hit_and_skips_overlap():
    index = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
    prices = pd.DataFrame(
        {
            "Open": [1.0, 1.0, 1.0, 1.0],
            "High": [1.0, 1.2, 1.0, 1.0],
            "Low": [1.0, 0.8, 1.0, 1.0],
            "Close": [1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )
    trades = pd.DataFrame(
        [
            {
                "Time": index[0],
                "Type": "LONG",
                "Entry": 1.0,
                "SL": 0.1,
                "TP": 0.1,
                "Size": 1.0,
            },
            {
                "Time": index[0],
                "Type": "LONG",
                "Entry": 1.0,
                "SL": 0.1,
                "TP": 0.1,
                "Size": 1.0,
            },
        ]
    )

    stats = run_backtest(trades, prices)

    assert stats["num_trades"] == 1
    assert stats["trades_df"].loc[0, "ExitReason"] == "SL"
    assert stats["trades_df"].loc[1, "Status"] == "SKIPPED_OVERLAP"
    assert stats["profit"] < 0


def test_schedule_and_capital_allocation_guards():
    saturday = datetime.datetime(2024, 1, 6, 12, tzinfo=datetime.UTC)
    friday_late = datetime.datetime(
        2024, 1, 5, 22, 30, tzinfo=datetime.UTC
    )
    assert is_market_closed(saturday)
    assert should_stop_new_trades(saturday)
    assert should_stop_new_trades(friday_late)

    allocation = allocate_capital(100, -1, 0, 0)
    assert allocation["trend_capital"] == allocation["range_capital"]
    assert allocation["range_capital"] == allocation["breakout_capital"]
    assert math.isclose(sum(allocation.values()), 100)

def test_live_loop_fetches_new_candles_before_deciding(monkeypatch):
    history = generate_sample_data(500)
    first_time = history.index[-1] + pd.Timedelta(minutes=1)
    second_time = first_time + pd.Timedelta(minutes=1)

    def candle_at(timestamp, close):
        return pd.DataFrame(
            {
                "Open": [close],
                "High": [close + 0.0001],
                "Low": [close - 0.0001],
                "Close": [close],
                "Volume": [100.0],
            },
            index=pd.DatetimeIndex([timestamp]),
        )

    class FakeBroker:
        def __init__(self):
            self.candle_calls = 0
            self.orders = []

        def get_recent_candles(self, lookback_minutes):
            self.candle_calls += 1
            return candle_at(
                first_time if self.candle_calls == 1 else second_time,
                1.2 + self.candle_calls * 0.0001,
            )

        def get_open_positions(self):
            return []

        def open_market_order(self, *args, **kwargs):
            self.orders.append((args, kwargs))
            return {"dealId": str(len(self.orders))}

    broker = FakeBroker()
    events = []
    monkeypatch.setattr(
        trade_engine,
        "initialize_ai_strategy",
        lambda history_df: events.append("initialize"),
    )
    monkeypatch.setattr(
        trade_engine,
        "refresh_ai_features",
        lambda candles: events.append(("refresh", candles.index[-1])),
    )
    monkeypatch.setattr(
        trade_engine,
        "should_stop_new_trades",
        lambda now: False,
    )
    monkeypatch.setattr(
        trade_engine,
        "generate_signal",
        lambda: (
            "BUY",
            {
                "signal_time": first_time,
                "entry_price": 1.2,
                "stop_pips": 10,
                "pip_value": 0.0001,
            },
        ),
    )
    monkeypatch.setattr(
        trade_engine.STOP_EVENT,
        "wait",
        lambda seconds: events.append(("wait", seconds)),
    )

    trade_engine.run_trading_bot(
        history_df=history,
        broker=broker,
        poll_interval=0,
        max_iterations=2,
    )

    assert broker.candle_calls == 2
    assert broker.orders and len(broker.orders) == 1
    assert events[:3] == [
        "initialize",
        ("refresh", first_time),
        ("wait", 0),
    ]
    assert ("refresh", second_time) in events
def test_monte_carlo_and_scenario_helpers_are_deterministic(tmp_path):
    result = run_monte_carlo([10, -5, 3], initial_equity=100, simulations=100)
    assert result["simulations"] == 100
    assert result["median_final_equity"] == 108
    prices = generate_sample_data(400)
    assert len(simulate_market_outage(prices, outage_bars=10)) == 390

    store = TradeStore(tmp_path / "state.sqlite3")
    store.record_decision("BUY", {"signal_time": "2024-01-01T00:00:00Z"})
    assert store.get_or_create_day_start("2024-01-01", 1000) == 1000
    assert store.get_or_create_day_start("2024-01-01", 900) == 1000


def test_walk_forward_keeps_train_before_test_and_scores_sharpe():
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    prices = pd.DataFrame({"Close": range(5)}, index=index)
    seen = []

    def trainer(train):
        seen.append(("train", train.index[-1]))
        return train.index[-1]

    def evaluator(model, test):
        seen.append(("test", test.index[0]))
        assert model < test.index[0]
        return {"passed": True}

    results = walk_forward(
        prices,
        train_window="1D",
        test_window="1D",
        trainer_fn=trainer,
        eval_fn=evaluator,
    )
    assert len(results) == 3
    assert probabilistic_sharpe_ratio([1, 2, 3]) > 0.5


def test_data_quality_drift_and_governance_guards():
    candles = generate_sample_data(400)
    candles.loc[candles.index[100], "High"] = -1
    cleaned = sanitize_candles(candles)
    assert len(cleaned) < len(candles)
    assert population_stability_index(pd.Series([1, 2, 3]), pd.Series([1, 2, 3])) == 0
    drift = adversarial_validation(
        create_pro_features(generate_sample_data(500)).iloc[:200],
        create_pro_features(generate_sample_data(500)).iloc[200:],
    )
    assert 0.0 <= drift["accuracy"] <= 1.0
    assert approve_trade(
        signal_present=True,
        risk_allowed=True,
        environment_safe=True,
        model_disagreement=0.1,
    ) == (True, "approved")
    assert approve_trade(
        signal_present=True,
        risk_allowed=True,
        environment_safe=True,
        model_disagreement=0.5,
    )[0] is False
