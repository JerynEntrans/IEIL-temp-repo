from __future__ import annotations

from shared.utils.db import Db
from .forecast_service import run_forecast


def handler(event, context):
    db = Db.from_env()
    try:
        return run_forecast(event, db=db)
    finally:
        db.close()
