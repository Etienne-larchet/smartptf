import polars as pl
import streamlit as st

from components.PageModels import RenderWarning
from models.Dpt import DPT
from models.Forecasting import Forecast
from models.Load import MarketIndex
from utils.polars import TimesSeriesPolars


class classproperty(property):
    def __get__(self, instance, owner):
        return self.fget(owner)


class MemorySession:
    @classproperty
    def marketindex(cls) -> MarketIndex:
        mktindex: MarketIndex = st.session_state.get("marketindex")
        if mktindex is None:
            raise RenderWarning("MarketIndex not found, please load the data.")
        return mktindex

    @classproperty
    def data_observation(cls) -> TimesSeriesPolars:
        if not (data_obs := st.session_state.get("data_observation")):
            delimitation_date = cls.marketindex.date_end - st.session_state["testing_period"]
            data_obs = TimesSeriesPolars(
                data=cls.marketindex.data.filter(pl.col("Date") <= delimitation_date), index_ticker=cls.index_ticker
            )
            st.session_state["data_observation"] = data_obs
        return data_obs

    @classproperty
    def data_testing(cls) -> TimesSeriesPolars:
        if not (data_obs := st.session_state.get("data_testing")):
            delimitation_date = cls.marketindex.date_end - st.session_state["testing_period"]
            data_obs = TimesSeriesPolars(
                data=cls.marketindex.data.filter(pl.col("Date") > delimitation_date), index_ticker=cls.index_ticker
            )
            st.session_state["data_testing"] = data_obs
        return data_obs

    @classproperty
    def dpt(cls) -> DPT:
        if not (dpt_obj := st.session_state.get("dpt")):
            index_ticker = cls.index_ticker
            dpt_obj = DPT(data=cls.data_observation.data, index_ticker=index_ticker)
            dpt_obj.calculate_signals()
            st.session_state["dpt"] = dpt_obj
        return dpt_obj

    @classproperty
    def index_ticker(cls) -> str:
        idx_ticker = st.session_state.get("index_ticker")
        if idx_ticker is None:
            raise RenderWarning("Please specify the index of the market index in the Import page.")
        return idx_ticker

    @classproperty
    def forecastor(cls) -> Forecast:
        if not (forecast_obj := st.session_state.get("forecastor")):
            index_ticker = cls.index_ticker
            data = cls.dpt.data
            forecast_obj = Forecast(index_ticker, data=data)
            st.session_state["forecastor"] = forecast_obj

        return forecast_obj
