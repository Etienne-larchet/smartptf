import os

import pytest
from dotenv import load_dotenv

from smartptf.dpt import DPT
from smartptf.Load import Indice
from smartptf.logging_config import configure_logging

load_dotenv()
configure_logging()


@pytest.fixture
def sp500_eo():
    sp500 = Indice(
        name="SP500",
        csv_compo_path="test/data/sp500_compo_until_2025-03-10.csv",
        date_end="2020-01-10",
        period="16y",
        eodhd_key=os.getenv("EODHD_API_KEY")
    )
    sp500.compo += ["GSPC.INDX"]
    sp500.load_from_eodhd(threshold_missing_val=0.001, display_progress=False)
    return sp500

@pytest.fixture
def sp500_csv():
    sp500 = Indice(
        name="SP500",
        csv_compo_path="test/data/sp500_compo_until_2025-03-10.csv",
        date_end="2020-01-10",
        period="16y",
    )
    sp500.load_from_csv()
    return sp500


def test_init(sp500_csv):
    dpt = DPT(sp500_csv.close, index_ticker="GSPC.INDX")
    assert dpt.log_returns is not None, "Log returns should be calculated successfully"
    assert len(dpt.log_returns) > 0, "Log returns DataFrame should not be empty"