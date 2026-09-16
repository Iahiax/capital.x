import pandas as pd


def analyze_daily(trades_df):
    if trades_df.empty:
        return pd.DataFrame(columns=['sum', 'count']).rename_axis('Date')

    trades_df = trades_df.copy()
    trades_df['Date'] = pd.to_datetime(trades_df['Time'], utc=True).dt.date
    grouped = trades_df.groupby('Date')
    stats = grouped['PnL'].agg(['sum', 'count'])
    return stats
