"""Run the trading research pipeline in live or safe sample-data mode,
and optionally run the live trading bot controlled via Telegram.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from backtest import run_backtest
from config import (
    INITIAL_EQUITY,
    RISK_PER_TRADE,
    WALK_FORWARD_TEST_DAYS,
    WALK_FORWARD_TRAIN_DAYS,
)
from daily_analyzer import analyze_daily
from data_loader import generate_sample_data, load_full_year_data
from drift_monitor import adversarial_validation
from features import create_pro_features
from logging_utils import configure_logging
from model import (
    add_ai_prob,
    generate_oof_ai_prob,
    train_meta_model,
    train_regime_models,
)
from monte_carlo import run_monte_carlo
from news_filter import build_news_blackout, fetch_forex_news
from optuna_optimize import run_optuna
from performance_metrics import stability_metrics
from readiness import validate_live_readiness
from service import run_continuous_service
from signals import generate_signals
from strategy import init_ai_model
from trade_engine import get_live_status, run_trading_bot
from walk_forward import run_ablation_report, validate_strategy_walk_forward

logger = logging.getLogger(__name__)

DEFAULT_SIGNAL_PARAMETERS = {
    "ai_long": 0.55,
    "ai_short": 0.45,
    "rvol_min": 1.2,
    "shock_max": 1.5,
    "noise_max": 1.5,
    "mq_min": 0.0,
    "score_min": 60.0,
}

# =========================
# 1) البحث التداولي (Pipeline)
# =========================

def research_main(sample: bool = False, trials: int = 5, monte_carlo_simulations: int = 1000):
    """
    تشغيل بايبلاين البحث التداولي:
    - توليد أو تحميل البيانات
    - بناء الميزات
    - تدريب نماذج Regime
    - تدريب Meta-Model
    - فلترة الأخبار
    - تشغيل Optuna
    - توليد إشارات
    - بناء صفقات
    - تشغيل باك تست
    - تحليل يومي
    """
    df = generate_sample_data() if sample else load_full_year_data()
    df_feat = create_pro_features(df)
    split = max(int(len(df_feat) * 0.7), 1)
    adversarial = adversarial_validation(df_feat.iloc[:split], df_feat.iloc[split:])
    logger.info("Adversarial validation: %s", adversarial)
    if adversarial["accuracy"] > 0.55:
        logger.warning(
            "Train/test period drift is high (accuracy %.3f).",
            adversarial["accuracy"],
        )

    validation = {
        "approved": False,
        "windows": [],
        "skipped": sample,
    }
    if not sample:
        validation = validate_strategy_walk_forward(
            df,
            train_window=f"{WALK_FORWARD_TRAIN_DAYS}D",
            test_window=f"{WALK_FORWARD_TEST_DAYS}D",
        )
        logger.info("Walk-forward validation: %s", validation)
        logger.info(
            "OOS aggregate: P&L=%.2f, max drawdown=%.2f, profit factor=%s, trades=%d",
            validation["aggregate"]["profit"],
            validation["aggregate"]["max_drawdown"],
            validation["aggregate"]["profit_factor"],
            validation["aggregate"]["num_trades"],
        )
        for regime, metrics in validation["regime_metrics"].items():
            logger.info(
                "OOS regime=%s: P&L=%.2f, max drawdown=%.2f, "
                "profit factor=%s, trades=%d",
                regime,
                metrics["profit"],
                metrics["max_drawdown"],
                metrics["profit_factor"],
                metrics["num_trades"],
            )

    oof_predictions = generate_oof_ai_prob(df_feat)
    models = train_regime_models(df_feat, persist=validation["approved"])
    df_feat = add_ai_prob(df_feat, models)
    meta_model = train_meta_model(
        df_feat,
        oof_predictions=oof_predictions,
        persist=validation["approved"],
    )

    df_news = fetch_forex_news()
    news_blackout = build_news_blackout(df, df_news)

    if sample:
        best_params = run_optuna(df_feat, df, n_trials=trials)
    else:
        # Optimizing on the complete history would leak future information
        # into the final research result. Use fixed defaults here; any tuning
        # for production must happen inside each walk-forward train window.
        best_params = DEFAULT_SIGNAL_PARAMETERS.copy()
        logger.info(
            "Skipping full-history parameter optimization; using fixed defaults "
            "after OOS validation."
        )
    logger.info("Optimized filters: %s", best_params)

    ablation_report = run_ablation_report(
        df,
        news_blackout=news_blackout,
        parameters=best_params,
        train_window=f"{WALK_FORWARD_TRAIN_DAYS}D",
        test_window=f"{WALK_FORWARD_TEST_DAYS}D",
        report_dir=Path(__file__).resolve().parent / "reports",
    )
    logger.info(
        "Ablation report: %d windows written to %s",
        ablation_report["window_count"],
        ablation_report.get("report_files", {}).get("json", "memory only"),
    )

    signals_df = generate_signals(
        df_feat,
        news_blackout=news_blackout,
        meta_model=meta_model,
        parameters=best_params,
    )

    atr = signals_df["ATR"].clip(lower=1e-6)
    score = signals_df["Score"]
    sl = atr * (1.0 - (score / 200.0).clip(upper=0.5))
    tp = atr * (1.0 + (score / 150.0).clip(upper=1.0))
    risk_amount = INITIAL_EQUITY * RISK_PER_TRADE * (score / 100).clip(0.5, 1.5)
    trades_df = pd.DataFrame(
        {
            "Time": signals_df["Time"],
            "Type": signals_df["Type"],
            "Entry": signals_df["Price"],
            "SL": sl,
            "TP": tp,
            "Size": risk_amount / sl,
            "Score": score,
        }
    )

    stats = run_backtest(trades_df, df)

    logger.info("In-sample diagnostic final equity: %s", stats["final_equity"])
    logger.info("In-sample diagnostic profit: %s", stats["profit"])
    logger.info("In-sample diagnostic win rate: %s", stats["win_rate"])
    logger.info("In-sample diagnostic profit factor: %s", stats["profit_factor"])
    logger.info("In-sample diagnostic max drawdown: %s", stats["max_drawdown"])
    logger.info(
        "In-sample diagnostic stability metrics: %s",
        stability_metrics(stats["trades_df"]),
    )
    if monte_carlo_simulations > 0:
        executed = stats["trades_df"].query("Status == 'EXECUTED'")
        logger.info(
            "Monte Carlo stress: %s",
            run_monte_carlo(
                executed["PnL"],
                initial_equity=INITIAL_EQUITY,
                simulations=monte_carlo_simulations,
            ),
        )

    daily_stats = analyze_daily(stats["trades_df"])
    logger.info("Daily performance:\n%s", daily_stats)


# =========================
# 2) نظام التداول الحي + تيليجرام
# =========================

def run_mode(mode: str):
    """
    يتم استدعاؤها من بوت تيليجرام عند تشغيل التداول.
    mode = DEMO أو LIVE
    """
    logger.info("بدء نظام التداول في وضع: %s", mode)

    # ضبط وضع التداول في config (ملاحظة: هذا تعديل في الذاكرة، وليس في الملف)
    import config
    if mode.upper() == "DEMO":
        config.USE_DEMO = True
    else:
        config.USE_DEMO = False

    # جلب بيانات سنة كاملة
    try:
        df = load_full_year_data()
        logger.info("تم تحميل البيانات التاريخية (%d شمعة)", len(df))
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("خطأ في جلب البيانات: %s", e)
        return

    if mode.upper() == "LIVE":
        try:
            validation = validate_strategy_walk_forward(
                df,
                train_window=f"{WALK_FORWARD_TRAIN_DAYS}D",
                test_window=f"{WALK_FORWARD_TEST_DAYS}D",
            )
            logger.info("نتيجة بوابة LIVE خارج العينة: %s", validation)
            validate_live_readiness(validation)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.critical("تم رفض تشغيل LIVE: %s", exc)
            return

    # تهيئة الذكاء الاصطناعي بنموذجك الفعلي
    try:
        init_ai_model(df)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("خطأ في تهيئة الذكاء الاصطناعي: %s", e)
        return

    # تشغيل نظام التداول الحي
    try:
        run_trading_bot(history_df=df)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("خطأ في نظام التداول: %s", e)


def stop_mode():
    """
    يتم استدعاؤها من بوت تيليجرام عند إيقاف التداول.
    يمكنك لاحقًا إضافة منطق لإغلاق الصفقات أو حفظ الحالة.
    """
    from trade_engine import request_stop

    request_stop()
    logger.info("تم إيقاف نظام التداول من تيليجرام.")


def telegram_main():
    """
    تشغيل بوت تيليجرام للتحكم في نظام التداول الحي.
    """
    from telegram_bot import start_telegram_bot

    logger.info("بدء بوت التحكم في نظام التداول عبر تيليجرام")
    start_telegram_bot(
        run_callback=run_mode,
        stop_callback=stop_mode,
        status_callback=get_live_status,
    )


# =========================
# 3) نقطة الدخول
# =========================

if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use deterministic local candles instead of calling Capital.com.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of Optuna trials (use 1 for a quick smoke test).",
    )
    parser.add_argument(
        "--live-bot",
        action="store_true",
        help="Run the live trading bot controlled via Telegram instead of the research pipeline.",
    )
    parser.add_argument(
        "--service",
        action="store_true",
        help="Run the continuous trading service instead of the one-shot research pipeline.",
    )
    parser.add_argument(
        "--service-mode",
        choices=("DEMO", "LIVE"),
        default="DEMO",
        help="Broker mode for --service; LIVE remains blocked by readiness gates.",
    )
    parser.add_argument(
        "--monte-carlo",
        type=int,
        default=1000,
        help="Number of trade-order Monte Carlo simulations (0 disables it).",
    )

    args = parser.parse_args()

    if args.service:
        run_continuous_service(mode=args.service_mode, sample=args.sample)
    elif args.live_bot:
        telegram_main()
    else:
        if args.trials < 1:
            parser.error("--trials must be at least 1")
        if args.monte_carlo < 0:
            parser.error("--monte-carlo cannot be negative")
        research_main(
            sample=args.sample,
            trials=args.trials,
            monte_carlo_simulations=args.monte_carlo,
        )
