from __future__ import annotations

from shared.utils.db import Db
from .training_service import run_goal_seek_training, run_training_register


def handler(event, context):
    """
    Lambda entry-point for training pipeline post-processing.

    Supported actions:
      - register: post-training SageMaker registration/deployment
      - train_goal_seek: inline goal-seek model training + metrics/predictions dump
    """
    action = event.get("action", "register")

    db = Db.from_env()
    try:
        if action == "register":
            return run_training_register(event, db=db)
        if action == "train_goal_seek":
            return run_goal_seek_training(event, db=db)
        raise ValueError(
            f"Unsupported action '{action}'. Supported actions: register, train_goal_seek."
        )
    finally:
        db.close()
