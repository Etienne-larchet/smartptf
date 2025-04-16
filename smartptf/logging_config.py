import json
from logging.config import dictConfig

from smartptf.config import ROOT_DIR


def configure_logging() -> None:
    LOGGING_CONFIG_FILE = ROOT_DIR / "logging_config.json"
    with open(LOGGING_CONFIG_FILE) as f:
        log_config = json.load(f)

    dictConfig(log_config)
