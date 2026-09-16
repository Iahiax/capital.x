# capital.x

Research and backtesting pipeline for Capital.com minute candles. It uses the
Capital.com session flow (`POST /api/v1/session`) and sends both `CST` and
`X-SECURITY-TOKEN` on subsequent requests.

## Safe local smoke test

The project can run without an account or network access:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python main.py --sample --trials 1
```

`--sample` uses deterministic synthetic candles and never contacts Capital.com.

The core requirements run with scikit-learn fallbacks. For the original
XGBoost/LightGBM ensemble and Optuna optimizer, also install
`requirements-optional.txt`.

## Live/demo data

1. Copy `.env.example` to `.env` or configure the same values as Replit
   environment secrets.
2. Set `CAPITAL_API_KEY`, `CAPITAL_IDENTIFIER`, and `CAPITAL_API_PASSWORD`.
   The password is the Capital.com API password, not necessarily the platform
   login password.
3. Keep `CAPITAL_USE_DEMO=true` until the strategy has been reviewed.
4. Run `python main.py --trials 5`.

Never commit `.env`, account credentials, API keys, or generated model files.
Credentials that were previously committed to a public repository should be
revoked and replaced at Capital.com and Finnhub before live use.

## Project layout

- `data_loader.py` — session creation, authenticated price requests, and sample data
- `features.py` / `orderflow.py` — technical and candle-derived features
- `model.py` / `meta_model.py` — regime models and decision layer
- `signals.py` / `backtest.py` — signal generation and historical simulation
- `main.py` — command-line entry point

