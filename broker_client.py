# broker_client.py

import requests
from config import API_KEY, EPIC, USE_DEMO
from session_manager import create_session

BASE_URL = (
    "https://demo-api-capital.backend-capital.com/api/v1"
    if USE_DEMO else
    "https://api-capital.backend-capital.com/api/v1"
)

class BrokerClient:
    def __init__(self):
        self.CST, self.XST = create_session()

    def _headers(self):
        return {
            "X-CAP-API-KEY": API_KEY,
            "CST": self.CST,
            "X-SECURITY-TOKEN": self.XST,
            "Content-Type": "application/json"
        }

    def get_open_positions(self):
        url = f"{BASE_URL}/positions"
        r = requests.get(url, headers=self._headers())
        if r.status_code != 200:
            print("❌ Error get_open_positions:", r.text)
            return []
        data = r.json()
        return data.get("positions", [])

    def open_market_order(self, direction, size, stop_loss=None, take_profit=None):
        url = f"{BASE_URL}/positions"
        body = {
            "epic": EPIC,
            "direction": direction,  # "BUY" أو "SELL"
            "size": size,
            "orderType": "MARKET"
        }
        if stop_loss is not None:
            body["stopLevel"] = stop_loss
        if take_profit is not None:
            body["limitLevel"] = take_profit

        r = requests.post(url, headers=self._headers(), json=body)
        print("📤 فتح صفقة:", r.status_code, r.text)
        return r.json() if r.status_code == 200 else None

    def close_position(self, deal_id):
        url = f"{BASE_URL}/positions/close"
        body = {
            "dealId": deal_id,
            "size": "ALL"
        }
        r = requests.post(url, headers=self._headers(), json=body)
        print("📤 إغلاق صفقة:", r.status_code, r.text)
        return r.json() if r.status_code == 200 else None
