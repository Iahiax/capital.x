def allocate_capital(equity, perf_trend, perf_range, perf_breakout):
    performances = [max(float(value), 0.0) for value in (
        perf_trend,
        perf_range,
        perf_breakout,
    )]
    if sum(performances) == 0:
        performances = [1.0, 1.0, 1.0]

    total = sum(performances)
    w_trend, w_range, w_breakout = (value / total for value in performances)

    return {
        'trend_capital': equity * w_trend,
        'range_capital': equity * w_range,
        'breakout_capital': equity * w_breakout
    }
