import pytest

from smartptf.dpt import Dpt
from smartptf.Load import Indice


@pytest.fixture
def sp500_close():
    sp500 = Indice(
        name="SP500",
        csv_compo_path="test/data/sp500_compo_until_2025-03-10.csv",
        date_end="2024-03-10",
        period="4y",
    )
    sp500.load_from_csv("test/data")
    return sp500.close

def test_init(sp500_close):
    dpt = Dpt(sp500_close)
    assert dpt.log_returns is not None, "Log returns should be calculated successfully"
    assert len(dpt.log_returns) > 0, "Log returns DataFrame should not be empty"