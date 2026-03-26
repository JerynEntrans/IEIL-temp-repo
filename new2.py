import os

import requests
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv(".env.deploy")


class ZohoTokenManager:
    def get_access_token(self) -> str:
        payload = {
            "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN"),
            "client_id": os.getenv("ZOHO_CLIENT_ID"),
            "client_secret": os.getenv("ZOHO_CLIENT_SECRET"),
            "grant_type": "refresh_token",
        }
        resp = requests.post(os.getenv("ZOHO_TOKEN_URL"), data=payload, timeout=30)
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
                        "Desalter Monitoring V2", "Boot Water Analysis Chloride", "Crude Details Crude Feed"
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

        resp = requests.post(os.getenv("ION_IOT_API_URL"), headers=self._headers, json=payload, timeout=60)
        resp.raise_for_status()

        if resp.status_code == 204 or not resp.text:
            return None
        return resp.json()

token = ZohoTokenManager().get_access_token()
client = ZohoIoTClient(token)

from_ts = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
to_ts = datetime(2026, 3, 2, 0, 0, 0, tzinfo=timezone.utc)

data = client.fetch_custom_range(
    plant_id="CDU1",
    from_ts=from_ts,
    to_ts=to_ts,
)
print(data)
