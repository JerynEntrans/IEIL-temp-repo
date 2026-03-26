import requests
import os
from datetime import datetime, timezone


class ZohoTokenManager:
    def get_access_token(self) -> str:
        payload = {
            "refresh_token": "1000.76a5ffb16dc832bdcd78141bd5af8a28.f7264ac0a6924e7abc53c4c27fb3c314",
            "client_id": "1000.K49DGH8QLIF147N7RGMLIZYOL9DHEC",
            "client_secret": "4c07721c121e47119ff9ffe1cc31cb1e2dd29e2469",
            "grant_type": "refresh_token",
        }
        resp = requests.post("https://accounts.zoho.in/oauth/v2/token", data=payload, timeout=30)
        resp.raise_for_status()
        token = resp.json().get("access_token")
        if not token:
            raise RuntimeError("Zoho access token not returned")
        return token


class ZohoIoTClient:
    def __init__(self, access_token: str):
        self._headers = {
            "Authorization": f"Zoho-oauthtoken {access_token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _fmt_ts(ts: datetime) -> str:
        return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    def fetch_custom_range(self, *, plant_id: str, from_ts: datetime, to_ts: datetime):
        payload = {
            "metrics": [
                {
                    "datapoint_names": [
                        "Desalter Monitoring V2"
                    ],
                    "instance_name": plant_id,
                    "period": "custom_range",
                    "custom_range": {
                        "from": {"value": self._fmt_ts(from_ts)},
                        "to": {"value": self._fmt_ts(to_ts)},
                    },
                    "sort_order": "asc",
                }
            ]
        }

        resp = requests.post("https://ionsitedevportal.ionexchange.in/iot/v1/datapoints/query/data", headers=self._headers, json=payload, timeout=60)
        resp.raise_for_status()

        if resp.status_code == 204 or not resp.text:
            return None
        return resp.json()
    

token_manager = ZohoTokenManager()
token = token_manager.get_access_token()
print("Access token:", token)
client = ZohoIoTClient(token)
from_ts = datetime(2025, 11, 1, tzinfo=timezone.utc)
to_ts = datetime(2026, 3, 2, tzinfo=timezone.utc)
data = client.fetch_custom_range(plant_id="CDU1", from_ts=from_ts, to_ts=to_ts)
print("Data:", data)
