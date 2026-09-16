# backtest.py

import pandas as pd
from config import INITIAL_EQUITY, MAX_LOOKAHEAD_MINUTES

def run_backtest(trades_df, df_prices):
    equity = INITIAL_EQUITY
    equity_curve = []
    wins = 0
    losses = 0

    for _, t in trades_df.iterrows():
        entry_time = t['Time']
        entry_price = t['Entry']
        tp = t['TP']
        sl = t['SL']
        size = t['Size']

        future = df_prices.loc[entry_time:].iloc[1:MAX_LOOKAHEAD_MINUTES]

        hit_tp = False
        hit_sl = False

        for _, bar in future.iterrows():
            high = bar['High']
            low = bar['Low']
            if t['Type'] == 'LONG':
                if high >= entry_price + tp:
                    hit_tp = True
                    break
                if low <= entry_price - sl:
                    hit_sl = True
                    break
            else:
                if low <= entry_price - tp:
                    hit_tp = True
                    break
                if high >= entry_price + sl:
                    hit_sl = True
                    break

        if hit_tp:
            profit = tp * size
            wins += 1
        elif hit_sl:
            profit = -sl * size
            losses += 1
        else:
            profit = 0

        equity += profit
        equity_curve.append(equity)

    num_trades = len(trades_df)
    win_rate = wins / num_trades * 100 if num_trades > 0 else 0

    return {
        'final_equity': equity,
        'profit': equity - INITIAL_EQUITY,
        'num_trades': num_trades,
        'win_rate': win_rate
    }
