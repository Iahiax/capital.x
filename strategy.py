# strategy.py
"""
واجهة الاستراتيجية الذكية:
- تهيئة نماذج الذكاء الاصطناعي باستخدام البيانات التاريخية
- استخدام نماذج Regime + Meta-Model لإنتاج إشارات تداول حية
"""

import pandas as pd
from typing import Tuple, Dict, Optional

from features import create_pro_features
from model import train_regime_models, add_ai_prob, train_meta_model, load_meta_model
from signals import generate_signals

AI_MODELS = {
    "regime_models": None,
    "meta_model": None,
}
LAST_FEATURES_DF: Optional[pd.DataFrame] = None


def init_ai_model(history_df: pd.DataFrame) -> None:
    """
    تهيئة نموذج الذكاء الاصطناعي:
    - بناء الميزات
    - تدريب نماذج Regime
    - إضافة احتمالات AI
    - تدريب Meta-Model
    - تحميله للاستخدام الحي
    """
    global AI_MODELS, LAST_FEATURES_DF

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

    AI_MODELS["regime_models"] = regime_models
    AI_MODELS["meta_model"] = meta_model
    LAST_FEATURES_DF = df_feat

    print("✅ تم تهيئة نماذج الذكاء الاصطناعي")


def generate_signal() -> Tuple[Optional[str], Dict]:
    """
    إنتاج إشارة تداول حية:
    - تستخدم آخر بيانات الميزات (LAST_FEATURES_DF)
    - تستخدم meta_model لتقييم الإشارة
    - تعيد:
        - signal: "BUY" / "SELL" / None
        - meta: dict يحتوي:
            - stop_pips
            - pip_value
            - regime
            - score
    """

    global AI_MODELS, LAST_FEATURES_DF

    if AI_MODELS["meta_model"] is None or LAST_FEATURES_DF is None:
        print("⚠️ نماذج الذكاء الاصطناعي غير مهيأة بعد – لا توجد إشارة")
        return None, {}

    # نفترض أن generate_signals يمكن أن يعمل على آخر صف واحد
    # أو على df_feat كامل ويعطي إشارات، نأخذ آخر إشارة
    signals_df = generate_signals(
        LAST_FEATURES_DF,
        news_blackout=None,
        meta_model=AI_MODELS["meta_model"],
    )

    if signals_df.empty:
        print("⚠️ لا توجد إشارات من النموذج")
        return None, {}

    last_signal = signals_df.iloc[-1]

    signal_type = last_signal["Type"]  # نفترض "BUY" أو "SELL"
    score = last_signal.get("Score", 0.0)
    atr = last_signal.get("ATR", 0.001)

    # منطق تحويل Score + ATR إلى stop_pips و pip_value
    stop_pips = atr * (1.0 - min(score / 200.0, 0.5)) * 10000
    pip_value = 0.0001

    meta = {
        "regime": last_signal.get("Regime", "UNKNOWN"),
        "stop_pips": max(stop_pips, 10.0),
        "pip_value": pip_value,
        "score": score,
    }

    return signal_type, meta
