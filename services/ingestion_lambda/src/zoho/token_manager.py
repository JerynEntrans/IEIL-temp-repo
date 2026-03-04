import os
import requests


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
