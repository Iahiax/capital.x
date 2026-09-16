# backtest.py

import pandas as pd

def run_backtest(trades):
    equity = 10000
    results = []

    for t in trades:
        if t['Type'] == 'LONG':
            profit = t['TP'] - t['Entry']
        else:
            profit = t['Entry'] - t['TP']

        equity += profit
        results.append(equity)

    return {
        'final_equity': equity,
        'profit': equity - 10000,
        'win_rate': 70  # placeholder
    }
