from __future__ import annotations

from shared.utils.db import Db
from .goal_seek_service import run_goal_seek


def handler(event, context):
    db = Db.from_env()
    try:
        return run_goal_seek(event, db=db)
    finally:
        db.close()
