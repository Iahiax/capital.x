# model.py

import xgboost as xgb
from lightgbm import LGBMClassifier
import joblib

def train_models(df):
    y = df['Target_3m']
    X = df.drop(columns=['Target_3m'])

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.02
    )
    xgb_model.fit(X_train, y_train)

    lgb_model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.02
    )
    lgb_model.fit(X_train, y_train)

    joblib.dump(xgb_model, "models/xgb_model.bin")
    joblib.dump(lgb_model, "models/lgb_model.bin")

    return xgb_model, lgb_model

def load_models():
    xgb_model = joblib.load("models/xgb_model.bin")
    lgb_model = joblib.load("models/lgb_model.bin")
    return xgb_model, lgb_model

def predict_probabilities(df, models):
    xgb_model, lgb_model = models
    X = df.drop(columns=['Target_3m'])
    df['AI_Prob'] = (xgb_model.predict_proba(X)[:,1] + lgb_model.predict_proba(X)[:,1]) / 2
    return df
