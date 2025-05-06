import os

import pytest
from dotenv import load_dotenv

from config.logging_config import configure_logging
from models.Dpt import DPT
from models.Load import Indice
from models.Prediction import Predict

load_dotenv("config/.env")
configure_logging()


@pytest.fixture
def sp500_eo():
    sp500 = Indice(
        name="SP500",
        csv_compo_path="test/data/sp500_compo_until_2025-03-10.csv",
        date_end="2020-01-10",
        period="16y",
        eodhd_key=os.getenv("EODHD_API_KEY"),
    )
    sp500.compo += ["GSPC.INDX"]
    sp500.load_from_eodhd(threshold_missing_val=0.001, display_progress=False)
    return sp500


@pytest.fixture
def sp500_csv():
    sp500 = Indice(
        name="SP500", csv_compo_path="test/data/sp500_compo_until_2025-03-10.csv", date_end="2020-01-10", period="16y"
    )
    sp500.load_from_csv()
    return sp500


def test_init(sp500_csv):
    dpt = DPT(sp500_csv.close, index_ticker="GSPC.INDX")
    dpt.calculate_signals()  # TODO add a control sin_theta**2 + cos_theta**2 = 1.0
    assert dpt.log_returns is not None, "Log returns should be calculated successfully"
    assert len(dpt.log_returns) > 0, "Log returns DataFrame should not be empty"

    val = (dpt.cos_theta.to_numpy() ** 2 + dpt.sin_theta.to_numpy() ** 2).flatten().sum()
    assert round(val, 3) == float(dpt.theta.shape[0] * dpt.theta.shape[1])


def test_solver(sp500_csv):
    dpt = DPT(sp500_csv.close, index_ticker="GSPC.INDX")
    dpt.calculate_signals()
    predictor = Predict(data=dpt.data)
    forecasts = predictor.moving_average()
    optimalpdf = dpt.solve(forecasts, C_alphas=0.8, C_betas=1.6)

    assert optimalpdf.weights is not None
    assert optimalpdf.returns is not None
    assert optimalpdf.betas is not None
    assert optimalpdf.alphas is not None
    assert optimalpdf.R is not None
    assert optimalpdf.ptf_return is not None
    assert optimalpdf.ptf_betas is not None
    assert optimalpdf.ptf_alphas is not None
