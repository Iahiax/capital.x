# config.py

EMAIL = "yahia.x@outlook.sa"                # بريد الحساب
PASSWORD = "Yahia-1411"             # كلمة مرور الحساب

API_KEY = "ut2RpxSbx6fiDdHv"              # مفتاح API
API_KEY_PASSWORD = "Yahia@1411"     # كلمة مرور مفتاح API

USE_DEMO = True           # الحساب التجريبي

EPIC = "CS.D.EURUSD.MINI.IP"
RESOLUTION = "MINUTE"

# رافعة مالية 100:1
LEVERAGE = 100

INITIAL_EQUITY = 10000
RISK_PER_TRADE = 0.005      # 0.5% من الحساب لكل صفقة

SPREAD = 0.0001
COMMISSION_PER_TRADE = 0.0

# أوقات السوق (مثال لفوركس)
MARKET_CLOSE_WEEKDAY = 4    # الجمعة
MARKET_CLOSE_HOUR = 23      # 23:00
MARKET_OPEN_WEEKDAY = 0     # الاثنين
MARKET_OPEN_HOUR = 0        # 00:00

# Finnhub News
FINNHUB_API_KEY = "d96hs19r01qr77dkhss0d96hs19r01qr77dkhssg"
NEWS_LOOKBACK_MINUTES = 60
NEWS_COOLDOWN_MINUTES = 30
