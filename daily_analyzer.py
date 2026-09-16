# daily_analyzer.py

import pandas as pd

def analyze_daily(trades_df):
    trades_df['Date'] = trades_df['Time'].dt.date
    grouped = trades_df.groupby('Date')
    stats = grouped['PnL'].agg(['sum', 'count'])
    return stats
