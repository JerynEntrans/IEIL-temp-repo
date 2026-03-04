from shared.utils.db import Db
from .ingestion_service import ingest_zoho_incremental


def handler(event, context):
    db = Db.from_env()

    try:
        result = ingest_zoho_incremental(event, db=db)
        db.close()
        return result
    except Exception:
        db.close()
        raise
