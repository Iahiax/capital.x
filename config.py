# config.py

EMAIL = "yahia.x@outlook.sa"
API_KEY = "ut2RpxSbx6fiDdHv"
API_KEY_PASSWORD = "Yahia@1411"
PASSWORD = "Yahia-1411"
USE_DEMO = True
EPIC = "EURUSD"
RESOLUTION = "MINUTE"

FULL_YEAR_MINUTES = 525600
BATCH_SIZE = 1500

INITIAL_EQUITY = 10000
RISK_PER_TRADE = 0.005      # 0.5% من الحساب لكل صفقة
MAX_LOOKAHEAD_MINUTES = 50  # عدد الدقائق لباك تست كل صفقة

SPREAD = 0.0001
COMMISSION_PER_TRADE = 0.0

# Finnhub News
FINNHUB_API_KEY = "d96hs19r01qr77dkhss0d96hs19r01qr77dkhssg"
NEWS_LOOKBACK_MINUTES = 60   # لا تداول قبل الأخبار بـ 60 دقيقة
NEWS_COOLDOWN_MINUTES = 30   # لا تداول بعد الأخبار بـ 30 دقيقة
