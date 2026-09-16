# alerts.py

def send_alerts(trades_df, stats):
    print("\n📢 Alerts / Summary:")
    print(f"Number of trades: {stats['num_trades']}")
    print(f"Win rate: {stats['win_rate']:.2f}%")
    print(f"Final equity: {stats['final_equity']:.2f}")
    print(f"Total profit: {stats['profit']:.2f}")
