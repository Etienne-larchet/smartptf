from dataclasses import dataclass

import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from utils.polars import TimesSeriesPolars


@dataclass
class Predict(TimesSeriesPolars):
    index_ticker: str = "GSPC.INDX"  # TODO Transform to variable

    def arima(self, auto: bool = True, order: tuple | None = None) -> pl.DataFrame:  # TODO graph the auto correlation
        if auto:
            model = AutoARIMA(season_length=12, trace=True)
        else:
            raise NotImplementedError("Setting ARIMA order is not yet implemented")

        sf = StatsForecast(models=[model], freq="1mo", n_jobs=-1, verbose=True)

        returns = self.get("logR", include_index=False, include_date=True)
        returns2 = returns.melt(id_vars="Date", variable_name="tickers", value_name="logR")

        model_fit = sf.fit(df=returns2, id_col="tickers", time_col="Date", target_col="logR")
        forecasts = model_fit.predict(1)
        forecasts2 = forecasts.pivot(values="AutoARIMA", index="Date", columns="tickers").drop("Date")
        return forecasts2

    def moving_average(self, window: int = 0) -> pl.DataFrame:
        returns = self.get("logR", include_index=False, include_date=False)
        return returns[-window:].mean()

    def exponential_smoothing(self):
        raise NotImplementedError("Exponential smoothing prediction is not implemented yet.")

    def lstm(self):
        raise NotImplementedError("LSTM prediction is not implemented yet.")
