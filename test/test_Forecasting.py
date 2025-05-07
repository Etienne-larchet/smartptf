import pytest

from config.logging_config import configure_logging
from models.Dpt import DPT
from models.Forecasting import Forecast
from models.Load import MarketIndex, MarkKetIndexComponents

configure_logging()


@pytest.fixture
def dpt():
    components = MarkKetIndexComponents(csv_compo_path="test/data/sp500_compo_until_2025-03-10.csv")
    compo = components.get_composition(date_ref="2020-01-01")
    sp500 = MarketIndex(name="SP500", compo=compo, date_end="2020-01-10", period="16y")
    sp500.load_from_csv()
    dpt = DPT(sp500.close, index_ticker="GSPC.INDX")
    return dpt


def test_moving_average(dpt):
    predictor = Forecast(data=dpt.data)
    predictions = predictor.moving_average(window=5)
    assert predictions is not None, "Moving average prediction should be calculated successfully"


def test_arima(dpt):
    predictor = Forecast(data=dpt.data)
    predictions = predictor.arima()
    assert predictions is not None, "Moving average prediction should be calculated successfully"
