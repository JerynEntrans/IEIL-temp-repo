from __future__ import annotations

import os

from shared.utils.db import Db

USE_ML_MODELS_RAW = os.getenv("USE_ML_MODELS")
if USE_ML_MODELS_RAW is None:
    raise ValueError("USE_ML_MODELS must be explicitly set to 'true' or 'false' for goal-seek lambda")

USE_ML_MODELS_VALUE = USE_ML_MODELS_RAW.strip().lower()
if USE_ML_MODELS_VALUE == "true":
    from .goal_seek_service import run_goal_seek
elif USE_ML_MODELS_VALUE == "false":
    from .no_ML_goal_seek_service import run_goal_seek
else:
    raise ValueError(f"Invalid USE_ML_MODELS value: {USE_ML_MODELS_RAW!r}. Expected 'true' or 'false'.")


def handler(event, context):
    db = Db.from_env()
    try:
        return run_goal_seek(event, db=db)
    finally:
        db.close()
