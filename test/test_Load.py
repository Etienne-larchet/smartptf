import os

import pytest

from smartptf.Load import Indice


@pytest.fixture
def sp500():
    return Indice(
        name="SP500",
        csv_compo_path="test/data/sp500_compo_until_2025-03-10.csv",
        date_end="2024-03-10",
        period="4y",
    )

def test_load_from_yahoo(sp500):
    sp500.load_from_yahoo()
    assert sp500._data is not None, "Data should be loaded successfully"
    assert len(sp500._data) > 0, "Data should not be empty"
    assert len(sp500.close) > 0, "Open data should not be empty"

def test_load_from_csv(sp500):
    sp500.load_from_csv('test/data')
    assert sp500._data is not None, "Data should be loaded successfully from CSV"
    assert len(sp500._data) > 0, "Data should not be empty after loading from CSV"
    assert len(sp500.open) > 0, "Open data should not be empty"

def test_get_composition(sp500):
    sp500.get_composition(date_ref="2025-03-10")
    assert sp500.compo is not None, "Composition should be retrieved successfully"
    assert len(sp500.compo) > 0, "Composition should not be empty"

def test_to_csv(sp500):
    sp500.load_from_yahoo()
    sp500.to_csv(directory='test/data')
    assert os.path.exists(sp500.csv_data_path), "CSV file should be created"


