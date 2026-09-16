# model.py

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from regime_engine import split_by_regime
from features import get_feature_columns
from meta_model import MetaDecisionModel
from config import MODEL_DIR

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


def train_regime_models(df_feat: pd.DataFrame):
    regimes = split_by_regime(df_feat)
    models = {}

    feature_cols = get_feature_columns(df_feat)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, dreg in regimes.items():
        if len(dreg) < 1000:
            continue

        X = dreg[feature_cols]
        y = dreg['Target_3m']
        if y.nunique() < 2:
            continue

        xgb = (
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
            )
            if XGBClassifier
            else HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=7
            )
        )

        lgb = (
            LGBMClassifier(
                n_estimators=200,
                max_depth=-1,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary",
                verbosity=-1,
            )
            if LGBMClassifier
            else HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=11
            )
        )

        xgb.fit(X, y)
        lgb.fit(X, y)

        models[name] = {'xgb': xgb, 'lgb': lgb}

        joblib.dump(xgb, MODEL_DIR / f"xgb_{name}.bin")
        joblib.dump(lgb, MODEL_DIR / f"lgb_{name}.bin")

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

        x = row[feature_cols].to_frame().T
        p1 = models[key]['xgb'].predict_proba(x)[0, 1]
        p2 = models[key]['lgb'].predict_proba(x)[0, 1]
        ai_probs.append((p1 + p2) / 2.0)

    df['AI_Prob'] = ai_probs
    return df


def train_meta_model(df_feat: pd.DataFrame) -> MetaDecisionModel:
    meta = MetaDecisionModel()
    meta.fit(df_feat)
    meta.save(MODEL_DIR / "meta_model.bin")
    return meta


def load_meta_model() -> MetaDecisionModel:
    meta = MetaDecisionModel()
    meta.load(MODEL_DIR / "meta_model.bin")
    return meta
