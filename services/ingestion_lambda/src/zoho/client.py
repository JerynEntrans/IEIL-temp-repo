import os
import requests
from datetime import datetime, timezone


class ZohoIoTClient:

    DATAPOINT_NAMES = [
        # crude
        "Crude Details Crude Feed",

        # desalter
        "Desalter Monitoring V2",
        "Desalter Monitoring Press",
        "Desalter Monitoring Interface Level",
        "Desalter 2 Monitoring Interface Level",

        # chemistry
        "Boot Water Analysis Chloride",
        "O/H Boot Water Analysis Chloride PPM",
    ]

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
                    "datapoint_names": self.DATAPOINT_NAMES,
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
