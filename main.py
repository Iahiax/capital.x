"""Run the trading research pipeline in live or safe sample-data mode,
and optionally run the live trading bot controlled via Telegram.
"""

import argparse

from config import INITIAL_EQUITY
from data_loader import generate_sample_data, load_full_year_data
from features import create_pro_features
from model import train_regime_models, add_ai_prob, train_meta_model, load_meta_model
from news_filter import fetch_forex_news, build_news_blackout
from signals import generate_signals
from risk_engine import RiskEngine
from backtest import run_backtest
from optuna_optimize import run_optuna
from daily_analyzer import analyze_daily

# إضافات النظام الحي + تيليجرام
from telegram_bot import start_telegram_bot
from trade_engine import run_trading_bot
from strategy import init_ai_model


# =========================
# 1) البحث التداولي (Pipeline)
# =========================

def research_main(sample: bool = False, trials: int = 5):
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

    models = train_regime_models(df_feat)
    df_feat = add_ai_prob(df_feat, models)

    meta_model = train_meta_model(df_feat)

    df_news = fetch_forex_news()
    news_blackout = build_news_blackout(df, df_news)

    best_params = run_optuna(df_feat, df, n_trials=trials)
    print("Optimized filters:", best_params)

    meta_model = load_meta_model()

    signals_df = generate_signals(df_feat, news_blackout=news_blackout, meta_model=meta_model)

    trades = []
    from config import RISK_PER_TRADE

    for _, s in signals_df.iterrows():
        atr = s["ATR"]
        score = s["Score"]

        sl = atr * (1.0 - min(score / 200.0, 0.5))
        tp = atr * (1.0 + min(score / 150.0, 1.0))

        base_risk = INITIAL_EQUITY * RISK_PER_TRADE
        quality_factor = min(max(score / 100.0, 0.5), 1.5)
        risk_amount = base_risk * quality_factor

        position_size = risk_amount / sl if sl > 0 else 0

        trades.append({
            "Time": s["Time"],
            "Type": s["Type"],
            "Entry": s["Price"],
            "SL": sl,
            "TP": tp,
            "Size": position_size,
            "Score": score,
        })

    import pandas as pd
    trades_df = pd.DataFrame(trades)

    stats = run_backtest(trades_df, df)

    print("Final equity:", stats["final_equity"])
    print("Profit:", stats["profit"])
    print("Win rate:", stats["win_rate"])
    print("Profit factor:", stats["profit_factor"])
    print("Max drawdown:", stats["max_drawdown"])

    daily_stats = analyze_daily(stats["trades_df"])
    print("Daily performance:")
    print(daily_stats)


# =========================
# 2) نظام التداول الحي + تيليجرام
# =========================

def run_mode(mode: str):
    """
    يتم استدعاؤها من بوت تيليجرام عند تشغيل التداول.
    mode = DEMO أو LIVE
    """
    print(f"🚀 بدء نظام التداول في وضع: {mode}")

    # ضبط وضع التداول في config (ملاحظة: هذا تعديل في الذاكرة، وليس في الملف)
    import config
    if mode.upper() == "DEMO":
        config.USE_DEMO = True
    else:
        config.USE_DEMO = False

    # جلب بيانات سنة كاملة
    try:
        df = load_full_year_data()
        print(f"📚 تم تحميل البيانات التاريخية ({len(df)} شمعة)")
    except Exception as e:
        print(f"❌ خطأ في جلب البيانات: {e}")
        return

    # تهيئة الذكاء الاصطناعي بنموذجك الفعلي
    try:
        init_ai_model(df)
    except Exception as e:
        print(f"❌ خطأ في تهيئة الذكاء الاصطناعي: {e}")
        return

    # تشغيل نظام التداول الحي
    try:
        run_trading_bot(history_df=df)
    except Exception as e:
        print(f"❌ خطأ في نظام التداول: {e}")


def stop_mode():
    """
    يتم استدعاؤها من بوت تيليجرام عند إيقاف التداول.
    يمكنك لاحقًا إضافة منطق لإغلاق الصفقات أو حفظ الحالة.
    """
    print("🛑 تم إيقاف نظام التداول من تيليجرام.")


def telegram_main():
    """
    تشغيل بوت تيليجرام للتحكم في نظام التداول الحي.
    """
    print("🤖 بدء بوت التحكم في نظام التداول عبر تيليجرام…")
    start_telegram_bot(run_callback=run_mode, stop_callback=stop_mode)


# =========================
# 3) نقطة الدخول
# =========================

if __name__ == "__main__":
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

    args = parser.parse_args()

    if args.live_bot:
        telegram_main()
    else:
        if args.trials < 1:
            parser.error("--trials must be at least 1")
        research_main(sample=args.sample, trials=args.trials)
