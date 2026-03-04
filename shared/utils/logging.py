import logging
import sys


def set_logging(log_name: str, log_level: int = logging.INFO):
    logging.basicConfig(format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    return logger
