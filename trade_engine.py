# trade_engine.py

import time
import datetime
from broker_client import BrokerClient
from scheduler import should_stop_new_trades
from strategy import generate_signal
from risk_manager import calculate_position_size
from config import INITIAL_EQUITY

def run_trading_bot(history_df=None):
    print("🤖 تشغيل نظام التداول بالذكاء الاصطناعي")

    broker = BrokerClient()
    equity = INITIAL_EQUITY

    # تمرير البيانات التاريخية للاستراتيجية الذكية (إن وجدت)
    if history_df is not None:
        print("📚 تمرير البيانات التاريخية إلى الاستراتيجية")
        initialize_ai_strategy(history_df=history_df)

    while True:
        now = datetime.datetime.utcnow()

        # منطق إيقاف فتح صفقات جديدة قبل إغلاق السوق
        if should_stop_new_trades(now):
            print("⏳ إيقاف فتح صفقات جديدة قبل إغلاق السوق")
            open_positions = broker.get_open_positions()
            if not open_positions:
                print("✅ لا توجد صفقات مفتوحة – البوت ينتظر حتى فتح السوق")
            else:
                print(f"📌 هناك {len(open_positions)} صفقة مفتوحة – مراقبتها فقط")
            time.sleep(60)
            continue

        # جلب آخر بيانات السوق (يمكنك ربطها بـ data_loader أو feed حي)
        # هنا نفترض أن الاستراتيجية نفسها تتولى قراءة البيانات اللحظية
        signal, meta = generate_signal()

        if signal:
            print(f"📈 إشارة من الذكاء الاصطناعي: {signal} | Meta: {meta}")

            stop_pips = meta.get("stop_pips", 20)
            pip_value = meta.get("pip_value", 0.0001)

            size = calculate_position_size(equity, stop_pips, pip_value)
            broker.open_market_order(signal, size)

        time.sleep(60)


def initialize_ai_strategy(history_df):
    """
    هذه الدالة اختيارية:
    تستخدم لتهيئة نموذج الذكاء الاصطناعي بالبيانات التاريخية
    (تدريب، تحميل وزنات، بناء سياق، إلخ)
    """
    from strategy import init_ai_model
    init_ai_model(history_df)
