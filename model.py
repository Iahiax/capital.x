# model.py

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from regime_engine import split_by_regime
from features import get_feature_columns
from meta_model import MetaDecisionModel


def train_regime_models(df_feat: pd.DataFrame):
    regimes = split_by_regime(df_feat)
    models = {}

    feature_cols = get_feature_columns(df_feat)

    for name, dreg in regimes.items():
        if len(dreg) < 1000:
            continue

        X = dreg[feature_cols]
        y = dreg['Target_3m']

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss'
        )

        lgb = LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary'
        )

        xgb.fit(X, y)
        lgb.fit(X, y)

        models[name] = {'xgb': xgb, 'lgb': lgb}

        joblib.dump(xgb, f"models/xgb_{name}.bin")
        joblib.dump(lgb, f"models/lgb_{name}.bin")

    return models


def add_ai_prob(df_feat: pd.DataFrame, models: dict) -> pd.DataFrame:
    df = df_feat.copy()
    feature_cols = get_feature_columns(df)

    ai_probs = []

    for idx, row in df.iterrows():
        regime = row['Regime']
        if regime == 1:
            key = 'uptrend'
        elif regime == -1:
            key = 'downtrend'
        elif regime == 0:
            key = 'range'
        else:
            key = 'chaos'

        if key not in models:
            ai_probs.append(0.5)
            continue

        x = row[feature_cols].values.reshape(1, -1)
        p1 = models[key]['xgb'].predict_proba(x)[0, 1]
        p2 = models[key]['lgb'].predict_proba(x)[0, 1]
        ai_probs.append((p1 + p2) / 2.0)

    df['AI_Prob'] = ai_probs
    return df


def train_meta_model(df_feat: pd.DataFrame) -> MetaDecisionModel:
    meta = MetaDecisionModel()
    meta.fit(df_feat)
    meta.save("models/meta_model.bin")
    return meta


def load_meta_model() -> MetaDecisionModel:
    meta = MetaDecisionModel()
    meta.load("models/meta_model.bin")
    return meta
