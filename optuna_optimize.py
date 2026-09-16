# optuna_optimize.py

import optuna
import pandas as pd
from risk import apply_risk_management
from backtest import run_backtest

def objective(trial, df_feat, df_prices):
    ai_long = trial.suggest_float("ai_long", 0.7, 0.9)
    ai_short = trial.suggest_float("ai_short", 0.1, 0.3)
    rvol_min = trial.suggest_float("rvol_min", 0.8, 1.5)
    shock_max = trial.suggest_float("shock_max", 1.0, 2.0)
    noise_max = trial.suggest_float("noise_max", 1.2, 2.0)

    signals = []

    for idx, row in df_feat.iterrows():
        hour = row['Hour']
        regime = row['Regime']

        if not (8 <= hour <= 11 or 14 <= hour <= 17):
            continue

        if regime == 2:
            continue

        if regime == 1:
            if (
                row['AI_Prob'] > ai_long and
                row['TrendStrength'] > 0 and
                row['Kalman_Fast_Slope'] > 0 and
                row['BuyPressure'] > 0.6 and
                row['RVOL'] > rvol_min and
                row['ShockIndex'] < shock_max and
                row['SmartDiv'] > 0 and
                row['NoiseIndex'] < noise_max
            ):
                signals.append({'Time': idx, 'Type': 'LONG', 'Price': row['Close'], 'ATR': row['ATR']})

        if regime == -1:
            if (
                row['AI_Prob'] < ai_short and
                row['TrendStrength'] < 0 and
                row['Kalman_Fast_Slope'] < 0 and
                row['SellPressure'] > 0.6 and
                row['RVOL'] > rvol_min and
                row['ShockIndex'] < shock_max and
                row['SmartDiv'] < 0 and
                row['NoiseIndex'] < noise_max
            ):
                signals.append({'Time': idx, 'Type': 'SHORT', 'Price': row['Close'], 'ATR': row['ATR']})

    signals_df = pd.DataFrame(signals)
    trades_df = apply_risk_management(signals_df)
    stats = run_backtest(trades_df, df_prices)

    # هدف Optuna: تعظيم final_equity وتقليل drawdown
    score = stats['final_equity'] - stats['max_drawdown']
    return score

def run_optuna(df_feat, df_prices, n_trials=30):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df_feat, df_prices), n_trials=n_trials)
    print("Best params:", study.best_params)
    print("Best score:", study.best_value)
    return study.best_params
