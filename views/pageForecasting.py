import plotly.express as px
import streamlit as st

from components.MemorySession import MemorySession as mm
from components.PageModels import PageModel, RenderWarning, StreamModel
from models.Forecasting import Forecast

MOVING_AVERAGE = "Moving Average"
ARIMA = "AutoArima"
LSTM = "LSTM"
ES = "Exponential Smoothing"


class DisplayForecast(StreamModel):
    name: str = "DefaultName"

    def __init__(self, window: int, method: str):
        self.window: int = window
        self.is_called: bool = True if method == self.name else False

        st.markdown(f"**{self.name}**")
        fig = px.bar()
        self.graph_pholder = st.empty()
        self.graph_pholder.plotly_chart(fig, key=self.name)

    def render_before(self):
        pass

    def render_after(self):
        pass


class DisplayMA(DisplayForecast):
    name = MOVING_AVERAGE

    def render(self):
        if self.is_called:
            mm.forecast_ma = mm.forecastor.moving_average(window=self.window, output="polars")
        if mm.forecast_ma is not None:
            fig = px.histogram(mm.forecast_ma, nbins=50)
            fig.update_layout(xaxis_tickformat=".1%", showlegend=False)
            self.graph_pholder.plotly_chart(fig)


class DisplayARIMA(DisplayForecast):
    name = ARIMA

    def render(self):
        if self.is_called:
            mm.forecast_arima = mm.forecastor.arima(output="polars")
        if mm.forecast_arima is not None:
            fig = px.histogram(mm.forecast_arima, nbins=50)
            fig.update_layout(xaxis_tickformat=".1%", showlegend=False)
            self.graph_pholder.plotly_chart(fig)


class DisplayES(DisplayForecast):
    name = ES

    def render(self):
        raise RenderWarning("Forecasting method not yet implemented.")


class DisplayLSTM(DisplayForecast):
    name = LSTM

    def render(self):
        raise RenderWarning("Forecasting method not yet implemented.")


# --- PAGE MASTER ---
class ForecastingPage(PageModel):
    def render(self):
        st.title("Forecasting 1 month")

        forecastor: Forecast = mm.forecastor
        col1, col2 = st.columns(2)
        val = mm.window if mm.window else len(forecastor.data)
        with col1:
            mm.window = st.slider("Observation window", min_value=1, max_value=len(forecastor.data), value=val)
        with col2:
            forecasting_method = st.pills("Forecasting method", [MOVING_AVERAGE, ARIMA, ES, LSTM])

        params = dict(window=mm.window, method=forecasting_method)
        col1, col2 = st.columns(2)
        with col1:
            DisplayMA(**params).run()
            DisplayES(**params).run()
        with col2:
            DisplayARIMA(**params).run()
            DisplayLSTM(**params).run()

        # TODO: Add a comparison with real returns


if __name__ == "__main__":
    ForecastingPage().run()
