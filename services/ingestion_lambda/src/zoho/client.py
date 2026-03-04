import os
import requests
from datetime import datetime


class ZohoIoTClient:
    def __init__(self, access_token: str):
        self._headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    def fetch_custom_range(self, *, plant_id: str, from_ts: datetime, to_ts: datetime) -> dict | None:
        payload = {
            "period": "custom_range",
            "custom_range": {"from": {"value": from_ts.isoformat()}, "to": {"value": to_ts.isoformat()}},
            "metrics": [{
                "datapoint_names": ["Desalter Monitoring V2", "Boot Water Analysis Chloride", "Crude Details Crude Feed"],
                "instance_name": plant_id,
                "sort_order": "asc",
            }],
        }

        resp = requests.post(os.getenv("ION_IOT_API_URL"), headers=self._headers, json=payload, timeout=60)
        resp.raise_for_status()

        if resp.status_code == 204 or not resp.text:
            return None
        return resp.json()
