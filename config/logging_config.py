import json
from logging.config import dictConfig


def configure_logging() -> None:
    LOGGING_CONFIG_FILE = "config/logging_config.json"
    with open(LOGGING_CONFIG_FILE) as f:
        log_config = json.load(f)

    dictConfig(log_config)
