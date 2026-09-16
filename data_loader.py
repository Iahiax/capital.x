# data_loader.py
# جلب البيانات من Capital.com بالطريقة الرسمية حسب التوثيق

import requests
import pandas as pd
import time
from config import (
    API_KEY,
    EMAIL,
    PASSWORD,
    EPIC,
    RESOLUTION,
    USE_DEMO
)

# اختيار بيئة Demo أو Live
BASE_URL = (
    "https://demo-api-capital.backend-capital.com/api/v1"
    if USE_DEMO else
    "https://api-capital.backend-capital.com/api/v1"
)


# ============================================================
# 1) إنشاء جلسة Session والحصول على CST و X-SECURITY-TOKEN
# ============================================================

def create_session():
    url = f"{BASE_URL}/session"

    headers = {
        "X-CAP-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "identifier": EMAIL,
        "password": PASSWORD,
        "encryptedPassword": False
    }

    r = requests.post(url, headers=headers, json=data)

    if r.status_code != 200:
        print("❌ خطأ في تسجيل الدخول:", r.text)
        raise Exception("فشل تسجيل الدخول إلى Capital.com")

    CST = r.headers.get("CST")
    XST = r.headers.get("X-SECURITY-TOKEN")

    if not CST or not XST:
        raise Exception("❌ لم يتم استلام CST أو X-SECURITY-TOKEN")

    print("✅ تم تسجيل الدخول بنجاح")
    print("CST:", CST)
    print("XST:", XST)

    return CST, XST


# ============================================================
# 2) جلب دفعة بيانات واحدة
# ============================================================

def fetch_batch(CST, XST, start, end):
    url = f"{BASE_URL}/prices/{EPIC}/{RESOLUTION}"

    headers = {
        "X-CAP-API-KEY": API_KEY,
        "CST": CST,
        "X-SECURITY-TOKEN": XST
    }

    params = {
        "from": start,
        "to": end,
        "pageSize": 1000
    }

    r = requests.get(url, headers=headers, params=params)

    if r.status_code != 200:
        print("❌ API Error:", r.text)
        return pd.DataFrame()

    data = r.json()

    if "prices" not in data:
        print("⚠️ لا يوجد حقل prices في الرد")
        return pd.DataFrame()

    rows = []
    for item in data["prices"]:
        if "snapshotTime" not in item:
            continue

        rows.append({
            "Time": pd.to_datetime(item["snapshotTime"]),
            "Open": item["openPrice"]["bid"],
            "High": item["highPrice"]["bid"],
            "Low": item["lowPrice"]["bid"],
            "Close": item["closePrice"]["bid"],
            "Volume": item.get("lastTradedVolume", 0)
        })

    return pd.DataFrame(rows)


# ============================================================
# 3) جلب سنة كاملة على دفعات أسبوعية
# ============================================================

def load_full_year_data():
    print("🚀 بدء جلب بيانات سنة كاملة من Capital.com")

    CST, XST = create_session()

    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=365)

    all_data = []

    current = start
    while current < end:
        batch_end = current + pd.Timedelta(days=7)

        print(f"📦 جلب البيانات من {current} إلى {batch_end}")

        df_batch = fetch_batch(
            CST, XST,
            current.strftime("%Y-%m-%dT%H:%M:%SZ"),
            batch_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        if df_batch.empty:
            print("⚠️ دفعة فارغة – قد يكون API رفض الطلب")
        else:
            all_data.append(df_batch)

        current = batch_end
        time.sleep(0.5)

    if len(all_data) == 0:
        raise Exception("❌ لم يتم جلب أي بيانات – تحقق من API أو المفاتيح")

    df = pd.concat(all_data, ignore_index=True)

    if "Time" not in df.columns:
        raise Exception("❌ خطأ: عمود Time غير موجود في البيانات")

    df = df.drop_duplicates().set_index("Time").sort_index()

    print("✅ تم تحميل البيانات بنجاح – عدد الشموع:", len(df))

    return df
