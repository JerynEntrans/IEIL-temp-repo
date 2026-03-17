from __future__ import annotations

import os

from shared.utils.db import Db

if os.getenv("USE_ML_MODELS", "false").lower() == "true":
    from .goal_seek_service import run_goal_seek
else:
    from .no_ML_goal_seek_service import run_goal_seek


def handler(event, context):
    db = Db.from_env()
    try:
        return run_goal_seek(event, db=db)
    finally:
        db.close()
