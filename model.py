# model.py

import xgboost as xgb
from lightgbm import LGBMClassifier
import joblib
from features import get_feature_columns

def train_models(df):
    y = df['Target_3m']
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    lgb_model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8
    )
    lgb_model.fit(X_train, y_train)

    joblib.dump(xgb_model, "models/xgb_model.bin")
    joblib.dump(lgb_model, "models/lgb_model.bin")

    return xgb_model, lgb_model

def load_models():
    xgb_model = joblib.load("models/xgb_model.bin")
    lgb_model = joblib.load("models/lgb_model.bin")
    return xgb_model, lgb_model

def add_ai_prob(df, models):
    xgb_model, lgb_model = models
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    prob_xgb = xgb_model.predict_proba(X)[:,1]
    prob_lgb = lgb_model.predict_proba(X)[:,1]
    df['AI_Prob'] = (prob_xgb + prob_lgb) / 2.0
    return df
