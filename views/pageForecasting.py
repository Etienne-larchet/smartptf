import streamlit as st

from components.MemorySession import MemorySession as mm
from components.PageModels import PageModel, RenderWarning, StreamModel
from models.Forecasting import Forecast


class DisplayForecast(StreamModel):
    def render_before(self):
        pass

    def render_after(self):
        pass


class DisplayMA(DisplayForecast):
    def render(self):
        forecastor: Forecast = mm.forecastor
        col1, _ = st.columns(2)
        with col1:
            window = st.slider(
                "Observation window", min_value=1, max_value=len(forecastor.data), value=len(forecastor.data)
            )
        closes = forecastor.get("logR").to_pandas()
        closes = closes.set_index("Date")
        st.write(closes)
        col, _ = st.columns([3, 1])
        from views.pageFourierAnalysis import set_graph

        set_graph(closes.T, "ok", col, "magma")
        # st.plotly_chart(fig)
        forecasts = forecastor.moving_average(window=window, output="polars")


class DisplayARIMA(DisplayForecast):
    def render(self):
        pass


class DisplayES(DisplayForecast):
    def render(self):
        raise RenderWarning("Forecasting method not yet implemented.")


class DisplayLSTM(DisplayForecast):
    def render(self):
        raise RenderWarning("Forecasting method not yet implemented.")


# --- PAGE MASTER ---
class ForecastingPage(PageModel):
    def render(self):
        st.title("Forecasting")
        forecasting_method = st.segmented_control(
            "Forecasting method", ["Moving Average", "ARIMA", "Exponential Smoothing", "LSTM"], default="Moving Average"
        )

        display_map = {
            "Moving Average": DisplayMA,
            "ARIMA": DisplayARIMA,
            "Exponential Smoothing": DisplayES,
            "LSTM": DisplayLSTM,
        }

        display_map[forecasting_method]().run()


if __name__ == "__main__":
    ForecastingPage().run()
