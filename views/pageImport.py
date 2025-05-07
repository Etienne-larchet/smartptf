import os
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import streamlit as st
from dateutil.relativedelta import relativedelta

from components.PageModels import PageModel, RenderWarning, StreamModel
from models.Load import MarketIndex, MarkKetIndexComponents
from utils.utils import Horizon, Period, relativedelta_str


@dataclass
class BaseDisplay(StreamModel):
    date_end: date
    period: relativedelta
    compo: list[str]
    eodhd_Key: str | None = None

    def render_before(self):
        pass

    def render_after(self):
        marketindex = st.session_state.get("marketindex")
        if marketindex is None:
            marketindex = MarketIndex(
                name="SP500", compo=self.compo, date_end=self.date_end, period=self.period, eodhd_key=self.eodhd_Key
            )
            st.session_state["marketindex"] = marketindex

        st.subheader("SP500 Data")
        if st.button("Load data", type="primary"):
            self.load(marketindex)
        if marketindex.data is None:
            raise RenderWarning("Press Load button.")
        else:
            data_selection = st.segmented_control(
                label="Data selection",
                options=["All", "Open", "High", "Low", "Close", "Volume"],
                default="All",
                label_visibility="collapsed",
            )
            match data_selection:
                case "Open":
                    data = marketindex.open
                case "High":
                    data = marketindex.high
                case "Low":
                    data = marketindex.low
                case "Close":
                    data = marketindex.close
                case "Volume":
                    data = marketindex.volume
                case "All":
                    data = marketindex.data
            st.dataframe(data, use_container_width=True)
        marketindex.to_csv()

    @abstractmethod
    def load(self, marketindex: MarketIndex): ...


class DisplayCSV(BaseDisplay):
    def render(self):
        pass

    def load(self, marketindex: MarketIndex):
        marketindex.load_from_csv()


class DisplayEODHD(BaseDisplay):
    threshold: float | None = field(init=False)

    def render(self):
        col1, col2 = st.columns([2, 1])
        with col1:
            self.eodhd_Key = st.text_input("EODHD API Key", type="password", value=os.getenv("EODHD_API_KEY"))
            if not self.eodhd_Key:
                raise RenderWarning("Please set your EODHD API key in the .env file or enter it above.")
        with col2:
            self.threshold = st.slider(
                "Threshold missing values (%)", min_value=0.0, max_value=0.1, step=0.01, value=0.0
            )

    def load(self, marketindex: MarketIndex):
        marketindex.load_from_eodhd(self.threshold)


class DisplayYahoo(BaseDisplay):
    threshold: float | None = field(init=False)

    def render(self):
        col, _ = st.columns([1, 3])
        with col:
            self.threshold = st.slider("Threshold missing values", min_value=0.0, max_value=0.1, step=0.01, value=0.0)
        raise RenderWarning("Yahoo is in process of bug fixing for now. Please select another method.")

    def load(self, marketindex: MarketIndex):
        marketindex.load_from_yahoo(self.threshold)


class DisplayAuto(BaseDisplay):
    def render(self):
        raise RenderWarning("Auto loading is not yet implemented. Please select another method.")

    def load(self, marketindex: MarketIndex):
        pass


# --- PAGE MASTER ---
class ImportPage(PageModel):
    def __init__(self):
        st.set_page_config(page_title="Loader", page_icon="📈", layout="wide")

    def render(self):
        st.title("Index Loader")
        st.subheader("Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            date_end = st.date_input("End Date", value="2025-01-01")
        with col2:
            observation_period = st.selectbox("Observation Period", Period.__args__, index=2)
            st.session_state["observation_period"] = observation_period
        with col3:
            testing_period = st.selectbox("Testing Period", Horizon.__args__, index=5)
            st.session_state["testing_period"] = testing_period

        st.subheader("SP500 Composition at date")
        st.session_state["market_ticker"] = st.text_input("Market ticker", value="GSPC.INDX")
        csv_compo_path = "./data/index_compo/sp500_compo_until_2025-03-10.csv"
        components = MarkKetIndexComponents(csv_path=csv_compo_path)
        compo_list = components.get_composition(date_end)
        compo = np.array(compo_list)
        compo = np.pad(compo, (0, -len(compo) % 15), constant_values=np.nan).reshape(-1, 15)
        st.dataframe(compo, use_container_width=True)

        load_method = st.segmented_control(
            "Select loading method", ["CSV", "EODHD", "YahooFinance", "Auto"], default="CSV"
        )

        period = relativedelta_str(observation_period) + relativedelta_str(testing_period)
        display_map = {"CSV": DisplayCSV, "EODHD": DisplayEODHD, "YahooFinance": DisplayYahoo, "Auto": DisplayAuto}
        display_map[load_method](date_end, period, compo_list).run()


if __name__ == "__main__":
    ImportPage().run()
