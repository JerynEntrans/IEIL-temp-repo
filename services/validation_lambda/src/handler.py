from __future__ import annotations

from shared.utils.db import Db
from .validation_service import run_validation


def handler(event, context):
    db = Db.from_env()
    try:
        return run_validation(event, db=db)
    finally:
        db.close()
