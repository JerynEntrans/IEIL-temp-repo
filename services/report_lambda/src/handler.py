from __future__ import annotations

from shared.utils.db import Db
from .report_service import generate_report


def handler(event, context):
    db = Db.from_env()
    try:
        return generate_report(event, db=db)
    finally:
        db.close()
