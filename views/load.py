import os
from datetime import date
from typing import Literal

import streamlit as st

from models.Load import Indice
from utils.utils import Period


@st.cache_data
def load_sp500(
    date_end: date,
    period: Period,
    csv_compo_path: str,
    load_method: Literal["CSV", "EODHD", "Yfinance", "Auto"],
    eodhd_api_key: str,
) -> Indice:
    sp500 = Indice(
        name="SP500", csv_compo_path=csv_compo_path, date_end=date_end, period=period, eodhd_key=eodhd_api_key
    )
    match load_method:
        case "CSV":
            sp500.load_from_csv()
        case "EODHD":
            sp500.load_from_eodhd(sp500)
        case "Yfinance":
            sp500.load_from_yahoo(sp500)
        case "Auto":
            st.warning("Auto loading is not yet implemented. Please select another method.")
    return sp500


if __name__ == "__main__":
    st.set_page_config(page_title="Loader", page_icon="📈", layout="wide")
    st.title("SP500 Loader")

    st.subheader("Parameters")
    col1, col2 = st.columns([1, 1])
    with col1:
        date_end = st.date_input("End Date", value="2020-01-01")
    with col2:
        period = st.selectbox("Period", Period.__args__, index=5)
    csv_compo_path = st.text_input("CSV Path", value="data/sp500_compo_until_2025-03-10.csv")

    load_method = st.radio("Select loading method", ["CSV", "EODHD", "Yfinance", "Auto"], horizontal=True)
    if load_method == "EODHD":
        eodhd_api_key = st.text_input("EODHD API Key", type="password", value=os.getenv("EODHD_API_KEY"))
        if not eodhd_api_key:
            st.warning("Please set your EODHD API key in the .env file or enter it above.")
    else:
        eodhd_api_key = None

    if st.button("Load data") or st.session_state.get("sp500") is not None:
        sp500 = load_sp500(date_end, period, csv_compo_path, load_method, eodhd_api_key)

        st.subheader("SP500 Composition at the end date")
        st.dataframe(sp500.compo, use_container_width=True)
        st.subheader("Loading Method")

        st.subheader("SP500 Data")
        if sp500._data is None:
            st.info("No data loaded yet. Please select a loading method and load the data.")
        else:
            data_selection = st.radio(
                label="Data selection",
                options=["All", "Open", "High", "Low", "Close", "Volume"],
                horizontal=True,
                label_visibility="collapsed",
            )
            match data_selection:
                case "Open":
                    data = sp500.open
                case "High":
                    data = sp500.high
                case "Low":
                    data = sp500.low
                case "Close":
                    data = sp500.close
                case "Volume":
                    data = sp500.volume
                case "All":
                    data = sp500._data
            st.dataframe(data, use_container_width=True)

        sp500.to_csv()
        st.session_state["sp500"] = sp500
