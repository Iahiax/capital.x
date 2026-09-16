# session_manager.py

import requests
from config import API_KEY, EMAIL, PASSWORD, USE_DEMO

BASE_URL = (
    "https://demo-api-capital.backend-capital.com/api/v1"
    if USE_DEMO else
    "https://api-capital.backend-capital.com/api/v1"
)

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
        print("❌ error.invalid.details:", r.text)
        raise Exception("فشل تسجيل الدخول")

    CST = r.headers.get("CST")
    XST = r.headers.get("X-SECURITY-TOKEN")

    if not CST or not XST:
        raise Exception("❌ لم يتم استلام الرموز")

    print("✅ تم تسجيل الدخول بنجاح")
    return CST, XST
