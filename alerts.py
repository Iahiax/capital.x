# alerts.py

def send_alerts(trades, stats):
    print("📢 Alerts:")
    print(f"Final Equity: {stats['final_equity']}")
    print(f"Profit: {stats['profit']}")
