"""Regime models, calibrated probabilities, and leakage-safe OOF training."""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from config import META_OOF_SPLITS, MODEL_DIR
from features import get_feature_columns
from meta_model import MetaDecisionModel
from regime_engine import split_by_regime

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


def _base_models() -> tuple:
    xgb = (
        XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=7,
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
            random_state=11,
        )
        if LGBMClassifier
        else HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=11
        )
    )
    return xgb, lgb


def _calibrated_fit(estimator, x: pd.DataFrame, y: pd.Series):
    """Fit isotonic probabilities, falling back safely for small folds."""

    try:
        try:
            calibrated = CalibratedClassifierCV(
                estimator=estimator, method="isotonic", cv=3
            )
        except TypeError:
            calibrated = CalibratedClassifierCV(
                base_estimator=estimator, method="isotonic", cv=3
            )
        calibrated.fit(x, y)
        return calibrated
    except (ValueError, RuntimeError):
        logger.warning("Probability calibration failed; using base classifier.")
        estimator.fit(x, y)
        return estimator


def _fit_regime_models(df_feat: pd.DataFrame, persist: bool) -> dict:
    regimes = split_by_regime(df_feat)
    models = {}
    feature_cols = get_feature_columns(df_feat)

    if persist:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, dreg in regimes.items():
        dreg = dreg[dreg["Target_3m"].notna()].copy()
        if len(dreg) < 100:
            continue

        x = dreg[feature_cols].replace([np.inf, -np.inf], np.nan)
        valid = ~x.isna().any(axis=1)
        x = x.loc[valid]
        y = dreg.loc[x.index, "Target_3m"].astype(int)
        if y.nunique() < 2:
            continue

        xgb, lgb = _base_models()
        xgb = _calibrated_fit(xgb, x, y)
        lgb = _calibrated_fit(lgb, x, y)
        models[name] = {"xgb": xgb, "lgb": lgb}

        if persist:
            joblib.dump(xgb, MODEL_DIR / f"xgb_{name}.bin")
            joblib.dump(lgb, MODEL_DIR / f"lgb_{name}.bin")

    return models


def train_regime_models(df_feat: pd.DataFrame, persist: bool = True):
    """Train the final regime models after validation has completed."""

    return _fit_regime_models(df_feat, persist=persist)


def _regime_key(regime: int) -> str:
    return {1: "uptrend", -1: "downtrend", 0: "range"}.get(regime, "chaos")


def add_ai_prob(df_feat: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Add ensemble probabilities without row-wise Python iteration."""

    df = df_feat.copy()
    feature_cols = get_feature_columns(df)
    ai_probs = pd.Series(0.5, index=df.index, dtype=float)
    disagreements = pd.Series(0.0, index=df.index, dtype=float)

    for regime, key in ((1, "uptrend"), (-1, "downtrend"), (0, "range"), (2, "chaos")):
        model_pair = models.get(key)
        if not model_pair:
            continue
        mask = df["Regime"].eq(regime)
        x = df.loc[mask, feature_cols].replace([np.inf, -np.inf], np.nan)
        valid = ~x.isna().any(axis=1)
        if not valid.any():
            continue
        p1 = model_pair["xgb"].predict_proba(x.loc[valid])[:, 1]
        p2 = model_pair["lgb"].predict_proba(x.loc[valid])[:, 1]
        ai_probs.loc[x.index[valid]] = (p1 + p2) / 2.0
        disagreements.loc[x.index[valid]] = np.abs(p1 - p2)

    df["AI_Prob"] = ai_probs
    df["AI_Disagreement"] = disagreements
    return df


def generate_oof_ai_prob(
    df_feat: pd.DataFrame,
    n_splits: int = META_OOF_SPLITS,
    embargo: int = 3,
) -> pd.Series:
    """Create out-of-fold regime probabilities for leakage-safe meta training."""

    target_rows = df_feat.index[df_feat["Target_3m"].notna()]
    result = pd.Series(np.nan, index=df_feat.index, dtype=float, name="AI_Prob")
    if len(target_rows) < 200 or n_splits < 2:
        raise ValueError("Not enough rows for out-of-fold meta-model training.")

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=embargo)
    positions = np.arange(len(target_rows))
    for train_pos, valid_pos in splitter.split(positions):
        train_index = target_rows[train_pos]
        valid_index = target_rows[valid_pos]
        fold_models = _fit_regime_models(
            df_feat.loc[train_index],
            persist=False,
        )
        fold_predictions = add_ai_prob(df_feat.loc[valid_index], fold_models)
        result.loc[valid_index] = fold_predictions["AI_Prob"]

    logger.info(
        "Generated OOF probabilities for %d/%d rows using %d folds.",
        result.notna().sum(),
        len(target_rows),
        n_splits,
    )
    scored = result.notna() & df_feat["Target_3m"].notna()
    if scored.any():
        logger.info(
            "OOF Brier score: %.5f",
            brier_score_loss(
                df_feat.loc[scored, "Target_3m"].astype(int),
                result.loc[scored],
            ),
        )
    return result


def train_meta_model(
    df_feat: pd.DataFrame,
    oof_predictions: pd.Series | None = None,
    persist: bool = True,
) -> MetaDecisionModel:
    """Train Meta-Model exclusively on out-of-fold base-model predictions."""

    if oof_predictions is None:
        oof_predictions = generate_oof_ai_prob(df_feat)
    training = df_feat.copy()
    training["AI_Prob"] = oof_predictions.reindex(training.index)
    meta = MetaDecisionModel()
    meta.fit(training)
    if persist:
        meta.save(MODEL_DIR / "meta_model.bin")
    return meta


def load_meta_model() -> MetaDecisionModel:
    meta = MetaDecisionModel()
    meta.load(MODEL_DIR / "meta_model.bin")
    return meta