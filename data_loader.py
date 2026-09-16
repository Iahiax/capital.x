# data_loader.py

import requests
import pandas as pd
from config import EMAIL, API_KEY, API_KEY_PASSWORD, DEMO, EPIC, RESOLUTION, FULL_YEAR_MINUTES, BATCH_SIZE

SERVER = "https://demo-api-capital.backend-capital.com" if DEMO else "https://api-capital.backend-capital.com"

def create_session():
    url = f"{SERVER}/api/v1/session"
    headers = {"X-CAP-API-KEY": API_KEY, "Content-Type": "application/json"}
    payload = {"identifier": EMAIL, "password": API_KEY_PASSWORD, "encryptedPassword": False}
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code == 200:
        return r.headers.get("CST"), r.headers.get("X-SECURITY-TOKEN")
    else:
        print("Login failed:", r.text)
        return None, None

def load_full_year_data():
    cst, xst = create_session()
    if not cst:
        raise Exception("Login failed")

    headers = {
        "X-CAP-API-KEY": API_KEY,
        "CST": cst,
        "X-SECURITY-TOKEN": xst
    }

    all_rows = []
    fetched = 0

    print("🚀 بدء جلب بيانات سنة كاملة لزوج EURUSD فريم دقيقة...")

    while fetched < FULL_YEAR_MINUTES:
        url = f"{SERVER}/api/v1/prices/{EPIC}"
        params = {
            "resolution": RESOLUTION,
            "max": BATCH_SIZE,
            "pageNumber": fetched // BATCH_SIZE
        }
        r = requests.get(url, headers=headers, params=params)
        if r.status_code != 200:
            print("Error:", r.text)
            break

        prices = r.json().get("prices", [])
        if not prices:
            break

        for p in prices:
            t = pd.to_datetime(p['snapshotTime'], unit='ms')
            all_rows.append({
                'Time': t,
                'Open': p['openPrice']['mid'],
                'High': p['highPrice']['mid'],
                'Low': p['lowPrice']['mid'],
                'Close': p['closePrice']['mid'],
                'Volume': p.get('lastTradedVolume', 100)
            })

        fetched += len(prices)
        print(f"📥 تم جلب {fetched} شمعة حتى الآن...")

    df = pd.DataFrame(all_rows).drop_duplicates().set_index('Time').sort_index()
    print(f"✅ تم جلب {len(df)} شمعة دقيقة تقريباً لسنة كاملة")
    return df
