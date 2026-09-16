# meta_model.py
# الطبقة العليا لاتخاذ القرار (Meta Decision Model)
# هذا النموذج يتعلم من:
# - AI_Prob (من نماذج XGB/LGB)
# - Score (درجة الإشارة)
# - Regime (وضع السوق)
# - MarketQuality (جودة السوق)
# ويقرر: هل ندخل الصفقة أم لا؟

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class MetaDecisionModel:
    def __init__(self):
        # نموذج بسيط لكنه قوي جداً في دمج الإشارات
        self.model = LogisticRegression(max_iter=500)

    def fit(self, df):
        """
        تدريب النموذج على بيانات الميزات النهائية.
        df يجب أن يحتوي على:
        - AI_Prob
        - Score
        - Regime
        - MarketQuality
        - Target_3m
        """

        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["AI_Prob", "Regime", "MarketQuality", "Target_3m"]
        )
        if df.empty:
            raise ValueError("Meta model has no complete training rows.")

        if "Score" not in df:
            df = df.copy()
            df["Score"] = self._score_frame(df)

        X = np.column_stack([
            df['AI_Prob'].values,
            df['Score'].values,
            df['Regime'].values,
            df['MarketQuality'].values
        ])

        y = df['Target_3m'].astype(int).values

        if np.unique(y).size < 2:
            raise ValueError("Meta model needs both positive and negative targets.")
        self.model.fit(X, y)

    def predict_prob(self, ai_prob, score, regime, mq):
        """
        يعطي احتمال الدخول في الصفقة بناءً على الطبقة العليا.
        """
        x = np.array([[ai_prob, score, regime, mq]])
        return self.model.predict_proba(x)[0, 1]

    def predict_prob_frame(self, ai_prob, score, regime, mq):
        x = np.column_stack([ai_prob, score, regime, mq])
        return self.model.predict_proba(x)[:, 1]

    def save(self, path="models/meta_model.bin"):
        import os

        os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path="models/meta_model.bin"):
        self.model = joblib.load(path)

    @staticmethod
    def _score_frame(df):
        is_short = df["Regime"] == -1
        directional_ai = np.where(is_short, 1 - df["AI_Prob"], df["AI_Prob"])
        directional_trend = np.where(is_short, -df["TrendStrength"], df["TrendStrength"])
        directional_pressure = np.where(
            is_short, df["SellPressure"], df["BuyPressure"]
        )
        directional_aggression = np.where(
            is_short, df["AggressiveSell"], df["AggressiveBuy"]
        )
        opposite_aggression = np.where(
            is_short, df["AggressiveBuy"], df["AggressiveSell"]
        )
        return (
            directional_ai * 50
            + (directional_trend / (df["ATR"] + 1e-6)) * 10
            + directional_pressure * 10
            + df["RVOL"] * 10
            - df["ShockIndex"] * 10
            - df["NoiseIndex"] * 10
            + directional_aggression * 5
            - opposite_aggression * 5
        )
