"""Distribution-shift and adversarial-validation checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Return PSI; values above .25 deserve review and above .40 stop trading."""

    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: tuple[str, ...] = (
        "ATR",
        "RVOL",
        "ShockIndex",
        "MarketTemperature",
    ),
) -> dict[str, float]:
    return {
        column: population_stability_index(reference[column], current[column])
        for column in columns
        if column in reference and column in current
    }


def adversarial_validation(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict:
    """Test whether a classifier can identify the time period of a row."""

    feature_columns = feature_columns or [
        column
        for column in reference.select_dtypes(include="number").columns
        if column in current and not column.startswith(("Target_", "Return_"))
    ]
    combined = pd.concat(
        [reference[feature_columns].assign(_period=0), current[feature_columns].assign(_period=1)]
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(combined) < 20 or combined["_period"].nunique() < 2:
        return {"accuracy": 0.5, "roc_auc": 0.5, "features": feature_columns}
    x = combined.drop(columns="_period")
    y = combined["_period"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=7),
    )
    model.fit(x, y)
    predicted = model.predict(x)
    probabilities = model.predict_proba(x)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "features": feature_columns,
    }


def page_hinkley(errors, delta: float = 0.005, threshold: float = 50.0) -> bool:
    """Lightweight Page-Hinkley alarm for a stream of model errors."""

    values = np.asarray(list(errors), dtype=float)
    if values.size < 10:
        return False
    mean = 0.0
    cumulative = 0.0
    minimum = 0.0
    for count, value in enumerate(values, start=1):
        mean += (value - mean) / count
        cumulative += value - mean - delta
        minimum = min(minimum, cumulative)
    return bool(cumulative - minimum > threshold)