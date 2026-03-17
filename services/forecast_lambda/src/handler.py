from __future__ import annotations

import os

from shared.utils.db import Db

if os.getenv("USE_ML_MODELS", "false").lower() == "true":
    from .forecast_service import run_forecast
else:
    from .no_ML_forcast_service import run_forecast


def handler(event, context):
    db = Db.from_env()
    try:
        return run_forecast(event, db=db)
    finally:
        db.close()
