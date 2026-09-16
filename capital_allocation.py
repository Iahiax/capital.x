# capital_allocation.py

def allocate_capital(equity, perf_trend, perf_range, perf_breakout):
    total = perf_trend + perf_range + perf_breakout + 1e-6
    w_trend = perf_trend / total
    w_range = perf_range / total
    w_breakout = perf_breakout / total

    return {
        'trend_capital': equity * w_trend,
        'range_capital': equity * w_range,
        'breakout_capital': equity * w_breakout
    }
