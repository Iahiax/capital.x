# capital.x

Research and backtesting pipeline for Capital.com minute candles. It uses the
Capital.com session flow (`POST /api/v1/session`) and sends both `CST` and
`X-SECURITY-TOKEN` on subsequent requests.

The project is research-first. The live trading loop is not a substitute for
broker-side validation, paper-trading, or independent risk controls.

## Safe local smoke test

The project can run without an account or network access:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python main.py --sample --trials 1
```

`--sample` uses deterministic synthetic candles and never contacts Capital.com.
It also works with no credentials, Telegram package, Finnhub key, or optional
machine-learning packages.

The core requirements run with scikit-learn fallbacks. For the original
XGBoost/LightGBM ensemble and Optuna optimizer, also install
`requirements-optional.txt`.

## Live/demo data

1. Configure the values in `.env.example` as process environment variables or
   Replit environment secrets. Do not commit `.env`.
2. Set `CAPITAL_API_KEY`, `CAPITAL_IDENTIFIER`, and `CAPITAL_API_PASSWORD`.
   The password is the Capital.com API password, not necessarily the platform
   login password.
3. Keep `CAPITAL_USE_DEMO=true` until the strategy has been reviewed.
4. Run `python main.py --trials 5`.

Telegram control additionally needs `TELEGRAM_BOT_TOKEN` and the optional
`python-telegram-bot` dependency. Reinforcement-learning experiments need the
optional `gymnasium` dependency.

Never commit `.env`, account credentials, API keys, or generated model files.
Credentials that were previously committed to a public repository should be
revoked and replaced at Capital.com and Finnhub before live use.

## Project layout

- `data_loader.py` — session creation, authenticated price requests, and sample data
- `features.py` / `orderflow.py` — technical and candle-derived features
- `model.py` / `meta_model.py` — regime models and decision layer
- `signals.py` / `backtest.py` — signal generation and historical simulation
- `walk_forward.py` — chronological validation and probabilistic Sharpe gate
- `monte_carlo.py` — trade-order reshuffling stress test
- `trade_store.py` — SQLite decision/order/equity audit trail
- `drift_monitor.py` — PSI, adversarial validation, and Page-Hinkley checks
- `data_quality.py` — OHLCV validation and outlier protection
- `governance.py` — signal, risk, and environment sign-off gate
- `performance_metrics.py` — stability metrics beyond final profit
- `risk_manager.py` / `risk_engine.py` — position sizing and drawdown guards
- `scheduler.py` / `news_filter.py` — market hours and news blackout guards
- `main.py` — command-line entry point

## Verification

```bash
ruff check .
pytest -q
python main.py --sample --trials 1 --monte-carlo 1000
```

The sample result is deterministic for the same dependency versions and is
only a pipeline smoke test, not evidence of profitability. On real historical
data, the pipeline runs chronological walk-forward validation before allowing
model persistence. Training rows inside the 3-minute label horizon are purged
from every fold, transaction costs are charged in the replay, and at least
three consecutive forward windows plus the aggregate out-of-sample result must
pass the profit/probabilistic-Sharpe gate. The report includes net P&L, maximum
drawdown, Profit Factor, and trade count for uptrend, downtrend, range, and
chaos regimes.

The Meta-Model is trained only from out-of-fold regime-model probabilities with
a time-series embargo. Base probabilities are isotonic-calibrated when the
fold has enough data. `Target_3m`, `Target_15m`, `Target_60m`, and a
cost-aware 3-minute label are retained for future multi-horizon experiments;
only `Target_3m` currently drives the production classifier.

## Continuous service

For a network-free continuous workflow check:

```bash
python main.py --service --sample
```

For broker demo mode, configure the Capital.com demo credentials and run:

```bash
python main.py --service --service-mode DEMO
```

The service fetches a fresh candle before every decision and remains alive until
SIGTERM/SIGINT. LIVE mode is intentionally blocked unless walk-forward
validation passes, `CAPITAL_USE_DEMO=false`, and `LIVE_TRADING_APPROVED=YES`.

### Replit workflow

The repository includes a safe continuous workflow command:

```text
cd .conversation/capital.x && python main.py --service --sample
```

For a broker-backed demo workflow, replace it with:

```text
cd .conversation/capital.x && python main.py --service --service-mode DEMO
```

Do not run the LIVE service as a substitute for the Telegram confirmation
flow. The recommended sequence is:

1. Run the research pipeline and review adversarial validation.
2. Run chronological walk-forward and paper/demo replay.
3. Keep `CAPITAL_USE_DEMO=true` while validating execution.
4. Use `/mode live`, then `/confirm_live`, only after the external review.
5. The process still refuses LIVE unless walk-forward is approved and
   `LIVE_TRADING_APPROVED=YES` is present.

To stop the continuous process, stop the Replit workflow or send SIGTERM.

### Safety notes

- Never put Capital.com, Telegram, or Finnhub credentials in GitHub.
- Use Replit Secrets or environment variables for credentials.
- `--sample` never calls a broker and cannot place orders.
- A missing, stale, or non-advancing candle feed prevents a new decision.
- Unknown open positions, severe feature drift, drawdown limits, and duplicate
  signals all block new orders.
- The sample pipeline is a correctness smoke test, not proof of profitability.

Market-state features now include soft trend probabilities, an independent
volatility regime, alternation/gap/autocorrelation microstructure measures, and
market temperature. These are diagnostics and model inputs; they do not turn
the system into a validated HMM or a low-latency market-data feed.

Before live use, run a paper replay and review the adversarial-validation
warning. An accuracy materially above 55% means the train/test periods are
easy to distinguish and the out-of-sample result needs investigation.
