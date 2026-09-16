# strategy.py
"""
استراتيجية التداول الحي بالذكاء الاصطناعي:
- تستخدم نفس البايبلاين البحثي:
    - create_pro_features
    - train_regime_models
    - add_ai_prob
    - train_meta_model
    - load_meta_model
    - generate_signals
- لكن بدل أن تنتهي بباك تست، تنتج إشارة حيّة:
    - BUY / SELL / None
    - مع Meta: stop_pips, pip_value, score, regime
"""

import pandas as pd
from typing import Tuple, Dict, Optional

from features import create_pro_features
from model import train_regime_models, add_ai_prob, train_meta_model, load_meta_model
from signals import generate_signals

AI_STATE = {
    "regime_models": None,
    "meta_model": None,
    "features_df": None,
}


def init_ai_model(history_df: pd.DataFrame) -> None:
    """
    تهيئة نموذج الذكاء الاصطناعي باستخدام البيانات التاريخية:
    - بناء الميزات
    - تدريب نماذج Regime
    - إضافة احتمالات AI
    - تدريب Meta-Model
    - تحميله للاستخدام الحي
    """
    global AI_STATE

    print("🧠 تهيئة نموذج الذكاء الاصطناعي بالبيانات التاريخية...")

    # بناء الميزات
    df_feat = create_pro_features(history_df)

    # تدريب نماذج Regime
    regime_models = train_regime_models(df_feat)

    # إضافة احتمالات AI
    df_feat = add_ai_prob(df_feat, regime_models)

    # تدريب Meta-Model
    meta_model = train_meta_model(df_feat)

    # تحميل Meta-Model (إذا كان محفوظًا)
    meta_model = load_meta_model()

    AI_STATE["regime_models"] = regime_models
    AI_STATE["meta_model"] = meta_model
    AI_STATE["features_df"] = df_feat

    print("✅ تم تهيئة نماذج الذكاء الاصطناعي للتداول الحي")


def generate_signal() -> Tuple[Optional[str], Dict]:
    """
    إنتاج إشارة تداول حيّة:
    - تستخدم آخر صف من df_feat
    - تستخدم meta_model لتقييم الإشارة
    - تعيد:
        - signal: "BUY" / "SELL" / None
        - meta: dict يحتوي:
            - stop_pips
            - pip_value
            - regime
            - score
    """

    global AI_STATE

    meta_model = AI_STATE["meta_model"]
    df_feat = AI_STATE["features_df"]

    if meta_model is None or df_feat is None or df_feat.empty:
        print("⚠️ نماذج الذكاء الاصطناعي غير مهيأة بعد – لا توجد إشارة")
        return None, {}

    # توليد الإشارات باستخدام نفس المنطق البحثي
    signals_df = generate_signals(
        df_feat,
        news_blackout=None,   # يمكنك لاحقًا ربط news_filter هنا
        meta_model=meta_model,
    )

    if signals_df.empty:
        print("⚠️ لا توجد إشارات من النموذج")
        return None, {}

    # نأخذ آخر إشارة كإشارة حيّة
    s = signals_df.iloc[-1]

    signal_type = s["Type"]          # "BUY" أو "SELL"
    score = s.get("Score", 0.0)
    atr = s.get("ATR", 0.001)
    regime = s.get("Regime", "UNKNOWN")

    # تحويل Score + ATR إلى stop_pips
    # نفس المنطق الذي كنت تستخدمه في main.py القديم تقريبًا
    sl = atr * (1.0 - min(score / 200.0, 0.5))
    stop_pips = max(sl * 10000, 10.0)   # تحويل إلى نقاط تقريبية
    pip_value = 0.0001

    meta = {
        "regime": regime,
        "stop_pips": stop_pips,
        "pip_value": pip_value,
        "score": score,
        "atr": atr,
    }

    return signal_type, meta
