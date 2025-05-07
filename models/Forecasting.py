from dataclasses import dataclass
from typing import Literal

import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from utils.polars import TimesSeriesPolars


@dataclass
class Forecast(TimesSeriesPolars):
    index_ticker: str

    def arima(
        self, auto: bool = True, order: tuple | None = None, output: Literal["polars", "dict"] = "dict"
    ) -> pl.DataFrame | dict[str, float]:  # TODO graph the auto correlation
        if auto:
            model = AutoARIMA(season_length=12, trace=True)
        else:
            raise NotImplementedError("Setting ARIMA order is not yet implemented")

        sf = StatsForecast(models=[model], freq="1mo", n_jobs=-1, verbose=True)

        returns = self.get("logR", include_index=False, include_date=True)
        returns2 = returns.melt(
            id_vars="Date", variable_name="tickers", value_name="logR"
        )  # TODO melt is depreciated, use unpivot

        model_fit = sf.fit(df=returns2, id_col="tickers", time_col="Date", target_col="logR")
        forecasts = model_fit.predict(1)
        forecasts2 = forecasts.pivot(values="AutoARIMA", index="Date", columns="tickers").drop("Date")
        if output == "polars":
            return forecasts2
        return forecasts2.to_dicts()[0]

    def moving_average(
        self, window: int = 0, output: Literal["polars", "dict"] = "dict"
    ) -> pl.DataFrame | dict[str, float]:
        returns = self.get("logR", include_index=False, include_date=False)
        mean = returns[-window:].mean()
        if output == "polars":
            return mean
        return mean.to_dicts()[0]

    def exponential_smoothing(self):
        raise NotImplementedError("Exponential smoothing prediction is not implemented yet.")

    def lstm(self):
        raise NotImplementedError("LSTM prediction is not implemented yet.")
