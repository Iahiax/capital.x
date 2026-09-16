# risk_engine.py

from config import INITIAL_EQUITY

class RiskEngine:
    def __init__(self, max_daily_dd=0.03, max_total_dd=0.2):
        self.equity_start = INITIAL_EQUITY
        self.max_daily_dd = max_daily_dd
        self.max_total_dd = max_total_dd
        self.total_dd = 0.0

    def update(self, equity):
        dd = (self.equity_start - equity) / self.equity_start
        self.total_dd = max(self.total_dd, dd)

    def can_trade(self, equity_today_start, equity_now):
        daily_dd = (equity_today_start - equity_now) / equity_today_start
        if daily_dd > self.max_daily_dd:
            return False
        if self.total_dd > self.max_total_dd:
            return False
        return True
