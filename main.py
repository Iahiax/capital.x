# main.py

from data_loader import load_full_year_data
from features import create_pro_features
from model import train_models, add_ai_prob
from signals import generate_signals
from risk import apply_risk_management
from backtest import run_backtest
from alerts import send_alerts
from optuna_optimize import run_optuna

def main():
    df = load_full_year_data()
    df_feat = create_pro_features(df)

    models = train_models(df_feat)
    df_feat = add_ai_prob(df_feat, models)

    best_params = run_optuna(df_feat, df, n_trials=20)
    print("✅ تم تحسين الفلاتر باستخدام Optuna")

    signals_df = generate_signals(df_feat)
    trades_df = apply_risk_management(signals_df)
    stats = run_backtest(trades_df, df)

    send_alerts(trades_df, stats)

if __name__ == "__main__":
    main()
