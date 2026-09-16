# trade_engine.py

import time
import datetime
from data_loader import load_full_year_data
from broker_client import BrokerClient
from strategy import generate_signal
from risk_manager import calculate_position_size
from scheduler import should_stop_new_trades
from config import INITIAL_EQUITY

def run_trading_bot():
    print("🚀 بدء نظام التداول")

    df = load_full_year_data()
    broker = BrokerClient()
    equity = INITIAL_EQUITY

    while True:
        now = datetime.datetime.utcnow()

        if should_stop_new_trades(now):
            print("⏳ إيقاف فتح صفقات جديدة قبل إغلاق السوق")
            open_positions = broker.get_open_positions()
            if not open_positions:
                print("✅ لا توجد صفقات مفتوحة – البوت ينتظر حتى فتح السوق")
            else:
                print(f"📌 هناك {len(open_positions)} صفقة مفتوحة – مراقبتها فقط")
            time.sleep(60)
            continue

        signal = generate_signal(df)

        if signal:
            print("📈 إشارة:", signal)
            stop_pips = 20
            pip_value = 0.0001
            size = calculate_position_size(equity, stop_pips, pip_value)
            broker.open_market_order(signal, size)

        time.sleep(60)
