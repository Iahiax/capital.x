from config import INITIAL_EQUITY


class RiskEngine:
    def __init__(self, max_daily_dd=0.03, max_total_dd=0.2):
        self.equity_start = INITIAL_EQUITY
        self.peak_equity = INITIAL_EQUITY
        self.max_daily_dd = max_daily_dd
        self.max_total_dd = max_total_dd
        self.total_dd = 0.0

    def update(self, equity):
        if self.peak_equity <= 0:
            self.total_dd = 1.0
            return
        self.peak_equity = max(self.peak_equity, equity)
        dd = max((self.peak_equity - equity) / self.peak_equity, 0.0)
        self.total_dd = max(self.total_dd, dd)

    def can_trade(self, equity_today_start, equity_now):
        if equity_today_start <= 0:
            return False
        daily_dd = (equity_today_start - equity_now) / equity_today_start
        return daily_dd <= self.max_daily_dd and self.total_dd <= self.max_total_dd
