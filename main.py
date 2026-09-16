# main.py

from data_loader import load_full_year_data
from features import create_pro_features
from model import train_models, load_models, predict_probabilities
from signals import generate_signals
from risk import apply_risk_management
from backtest import run_backtest
from alerts import send_alerts

def main():
    df = load_full_year_data()
    df = create_pro_features(df)

    models = train_models(df)
    df = predict_probabilities(df, models)

    signals = generate_signals(df)
    trades = apply_risk_management(signals)

    stats = run_backtest(trades)
    send_alerts(trades, stats)

if __name__ == "__main__":
    main()
