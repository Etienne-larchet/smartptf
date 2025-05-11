import os

import pytest
from dotenv import load_dotenv

from config.logging_config import configure_logging
from models.Load import MarketIndex, MarkKetIndexComponents

load_dotenv("config/.env")
configure_logging()


@pytest.fixture
def sp500():
    components = MarkKetIndexComponents(csv_compo_path="data/index_compo/sp500_compo_until_2025-03-10.csv")
    compo = components.get_composition(date_ref="2020-01-01")
    return MarketIndex(
        name="SP500", compo=compo, date_end="2020-01-01", period="16y", eodhd_key=os.getenv("EODHD_API_KEY")
    )


def test_load_from_yahoo(sp500):
    sp500.load_from_yahoo()
    assert sp500._data is not None, "Data should be loaded successfully"
    assert len(sp500._data) > 0, "Data should not be empty"
    assert len(sp500.close) > 0, "Close data should not be empty"


def test_load_from_csv(sp500):
    sp500.load_from_csv("test/data")
    assert sp500._data is not None, "Data should be loaded successfully from CSV"
    assert len(sp500._data) > 0, "Data should not be empty after loading from CSV"
    assert len(sp500.open) > 0, "Open data should not be empty"


def test_load_from_eodhd(sp500):
    sp500.load_from_eodhd()
    assert sp500._data is not None, "Data should be loaded successfully from EODHD"
    assert len(sp500._data) > 0, "Data should not be empty after loading from EODHD"
    assert len(sp500.high) > 0, "High data should not be empty"


def test_get_composition(sp500):
    sp500.get_composition(date_ref="2025-03-10")
    assert sp500.compo is not None, "Composition should be retrieved successfully"
    assert len(sp500.compo) > 0, "Composition should not be empty"


def test_to_csv(sp500):
    sp500.load_from_eodhd(threshold_missing_val=0.01)
    sp500.to_csv(directory="test/data")
    assert os.path.exists(sp500.csv_data_path), "CSV file should be created"
