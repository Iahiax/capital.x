# meta_model.py
# الطبقة العليا لاتخاذ القرار (Meta Decision Model)
# هذا النموذج يتعلم من:
# - AI_Prob (من نماذج XGB/LGB)
# - Score (درجة الإشارة)
# - Regime (وضع السوق)
# - MarketQuality (جودة السوق)
# ويقرر: هل ندخل الصفقة أم لا؟

import numpy as np
import joblib
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

        X = np.column_stack([
            df['AI_Prob'].values,
            df['Score'].values,
            df['Regime'].values,
            df['MarketQuality'].values
        ])

        y = df['Target_3m'].values

        self.model.fit(X, y)

    def predict_prob(self, ai_prob, score, regime, mq):
        """
        يعطي احتمال الدخول في الصفقة بناءً على الطبقة العليا.
        """
        x = np.array([[ai_prob, score, regime, mq]])
        return self.model.predict_proba(x)[0, 1]

    def save(self, path="models/meta_model.bin"):
        joblib.dump(self.model, path)

    def load(self, path="models/meta_model.bin"):
        self.model = joblib.load(path)
