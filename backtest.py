# backtest.py

import pandas as pd
from config import INITIAL_EQUITY, MAX_LOOKAHEAD_MINUTES

def run_backtest(trades_df, df_prices, spread=0.0001, commission_per_trade=0.0):
    equity = INITIAL_EQUITY
    equity_curve = []
    wins = 0
    losses = 0
    profits = []
    losses_list = []

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
                if high >= entry_price + tp + spread:
                    hit_tp = True
                    break
                if low <= entry_price - sl - spread:
                    hit_sl = True
                    break
            else:
                if low <= entry_price - tp - spread:
                    hit_tp = True
                    break
                if high >= entry_price + sl + spread:
                    hit_sl = True
                    break

        if hit_tp:
            profit = tp * size - commission_per_trade
            wins += 1
            profits.append(profit)
        elif hit_sl:
            profit = -sl * size - commission_per_trade
            losses += 1
            losses_list.append(-profit)
        else:
            profit = -commission_per_trade

        equity += profit
        equity_curve.append(equity)

    num_trades = len(trades_df)
    win_rate = wins / num_trades * 100 if num_trades > 0 else 0

    total_profit = sum(profits)
    total_loss = sum(losses_list) if len(losses_list) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else None

    max_equity = INITIAL_EQUITY
    max_dd = 0
    for e in equity_curve:
        if e > max_equity:
            max_equity = e
        dd = max_equity - e
        if dd > max_dd:
            max_dd = dd

    return {
        'final_equity': equity,
        'profit': equity - INITIAL_EQUITY,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd
    }
