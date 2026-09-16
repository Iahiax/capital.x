# optuna_optimize.py

import optuna
import pandas as pd
from signals import compute_signal_score
from risk_engine import RiskEngine
from backtest import run_backtest


def objective(trial, df_feat, df_prices):
    ai_long = trial.suggest_float("ai_long", 0.7, 0.9)
    ai_short = trial.suggest_float("ai_short", 0.1, 0.3)
    rvol_min = trial.suggest_float("rvol_min", 0.8, 1.5)
    shock_max = trial.suggest_float("shock_max", 1.0, 2.0)
    noise_max = trial.suggest_float("noise_max", 1.2, 2.0)
    mq_min = trial.suggest_float("mq_min", -0.2, 0.3)
    score_min = trial.suggest_float("score_min", 50, 80)

    signals = []

    for idx, row in df_feat.iterrows():
        hour = row['Hour']
        regime = row['Regime']
        mq = row['MarketQuality']

        if not (8 <= hour <= 11 or 14 <= hour <= 17):
            continue

        if mq < mq_min:
            continue

        if regime == 2:
            continue

        score = compute_signal_score(row)

        if score < score_min:
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
                signals.append({'Time': idx, 'Type': 'LONG', 'Price': row['Close'], 'ATR': row['ATR'], 'Score': score})

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
                signals.append({'Time': idx, 'Type': 'SHORT', 'Price': row['Close'], 'ATR': row['ATR'], 'Score': score})

    signals_df = pd.DataFrame(signals)

    from risk_engine import RiskEngine
    from config import INITIAL_EQUITY, RISK_PER_TRADE

    trades = []
    for _, s in signals_df.iterrows():
        atr = s['ATR']
        score = s['Score']

        sl = atr * (1.0 - min(score / 200.0, 0.5))
        tp = atr * (1.0 + min(score / 150.0, 1.0))

        base_risk = INITIAL_EQUITY * RISK_PER_TRADE
        quality_factor = min(max(score / 100.0, 0.5), 1.5)
        risk_amount = base_risk * quality_factor

        position_size = risk_amount / sl if sl > 0 else 0

        trades.append({
            'Time': s['Time'],
            'Type': s['Type'],
            'Entry': s['Price'],
            'SL': sl,
            'TP': tp,
            'Size': position_size,
            'Score': score
        })

    trades_df = pd.DataFrame(trades)
    stats = run_backtest(trades_df, df_prices)

    score_obj = stats['final_equity'] - stats['max_drawdown']
    return score_obj


def run_optuna(df_feat, df_prices, n_trials=30):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df_feat, df_prices), n_trials=n_trials)
    print("Best params:", study.best_params)
    print("Best score:", study.best_value)
    return study.best_params
