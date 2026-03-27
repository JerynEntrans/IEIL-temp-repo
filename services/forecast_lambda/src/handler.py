from __future__ import annotations

import os

from shared.utils.db import Db

USE_ML_MODELS_RAW = os.getenv("USE_ML_MODELS")
if USE_ML_MODELS_RAW is None:
    raise ValueError("USE_ML_MODELS must be explicitly set to 'true' or 'false' for forecast lambda")

USE_ML_MODELS_VALUE = USE_ML_MODELS_RAW.strip().lower()
if USE_ML_MODELS_VALUE == "true":
    from .forecast_service import run_forecast
elif USE_ML_MODELS_VALUE == "false":
    from .no_ML_forcast_service import run_forecast
else:
    raise ValueError(f"Invalid USE_ML_MODELS value: {USE_ML_MODELS_RAW!r}. Expected 'true' or 'false'.")


def handler(event, context):
    db = Db.from_env()
    try:
        return run_forecast(event, db=db)
    finally:
        db.close()
