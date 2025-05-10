from collections import defaultdict

import polars as pl
import streamlit as st

from components.PageModels import RenderWarning
from models.DPT.DptCls import DPT, OptimizedPortfolio
from models.Forecasting import Forecast
from models.Load import MarketIndex
from utils.polars import TimesSeriesPolars


class MemorySessionCls:
    def __setattr__(self, name, value):
        st.session_state[name] = value

    @property
    def marketindex(self) -> MarketIndex:
        mktindex: MarketIndex = st.session_state.get("marketindex")
        if mktindex is None:
            raise RenderWarning("MarketIndex not found, please load the data.")
        return mktindex

    @property
    def data_observation(self) -> TimesSeriesPolars:
        if not (data_obs := st.session_state.get("data_observation")):
            delimitation_date = self.marketindex.date_end - st.session_state["testing_period"]
            data_obs = TimesSeriesPolars(
                data=self.marketindex.data.filter(pl.col("Date") <= delimitation_date), index_ticker=self.index_ticker
            )
            st.session_state["data_observation"] = data_obs
        return data_obs

    @property
    def data_testing(self) -> TimesSeriesPolars:
        if not (data_obs := st.session_state.get("data_testing")):
            delimitation_date = self.marketindex.date_end - st.session_state["testing_period"]
            data_obs = TimesSeriesPolars(
                data=self.marketindex.data.filter(pl.col("Date") > delimitation_date), index_ticker=self.index_ticker
            )
            st.session_state["data_testing"] = data_obs
        return data_obs

    @property
    def dpt(self) -> DPT:
        if not (dpt_obj := st.session_state.get("dpt")):
            index_ticker = self.index_ticker
            dpt_obj = DPT(data=self.data_observation.data, index_ticker=index_ticker)
            dpt_obj.calculate_signals()
            st.session_state["dpt"] = dpt_obj
        return dpt_obj

    @property
    def index_ticker(self) -> str:
        idx_ticker = st.session_state.get("index_ticker")
        if idx_ticker is None:
            raise RenderWarning("Please specify the index of the market index in the Import page.")
        return idx_ticker

    @property
    def forecastor(self) -> Forecast:
        if not (forecast_obj := st.session_state.get("forecastor")):
            index_ticker = self.index_ticker
            data = self.dpt.data
            forecast_obj = Forecast(index_ticker, data=data)
            st.session_state["forecastor"] = forecast_obj

        return forecast_obj

    @property
    def forecast_ma(self) -> pl.DataFrame | None:
        return st.session_state.get("forecast_ma")

    @property
    def forecast_arima(self) -> pl.DataFrame | None:
        return st.session_state.get("forecast_arima")

    @property
    def forecast_es(self) -> pl.DataFrame | None:
        return st.session_state.get("forecast_es")

    @property
    def forecast_lstm(self) -> pl.DataFrame | None:
        return st.session_state.get("forecast_lstm")

    @property
    def window(self) -> int:
        return st.session_state.get("window")

    @property
    def alpha(self) -> defaultdict:
        return st.session_state.get("alpha", defaultdict(lambda: 0.5))

    @property
    def beta(self) -> defaultdict:
        return st.session_state.get("beta", defaultdict(lambda: 0.5))

    @property
    def ptf_size(self) -> int:
        return st.session_state.get("ptf_size", 15)

    @property
    def min_invest(self) -> float:
        return st.session_state.get("min_invest", 0.02)

    @property
    def max_invest(self) -> float:
        return st.session_state.get("max_invest", 0.2)

    @property
    def optptf(self) -> OptimizedPortfolio:
        return st.session_state.get("optptf")

    @property
    def advancedcontrols_signals(self) -> defaultdict:
        return st.session_state.get("advancedcontrols_signals", defaultdict(lambda: False))


MemorySession = MemorySessionCls()
