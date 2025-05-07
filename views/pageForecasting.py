import streamlit as st

from components.PageModels import PageModel, RenderWarning


class ForecastingPage(PageModel):
    def render(self):
        st.title("Forecasting")
        raise RenderWarning("Not yet implemented.")


if __name__ == "__main__":
    ForecastingPage().run()
