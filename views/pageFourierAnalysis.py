import numpy as np
import plotly.express as px
import streamlit as st

from components.PageModels import PageModel, RenderWarning
from models.Dpt import DPT
from models.Load import MarketIndex


def set_graph(data: np.ndarray, title: str, column: st._DeltaGenerator, colormap: str) -> None:
    subcol1, subcol2, _ = column.columns(3)
    with subcol1:
        quantile_min = st.slider("Zmin", value=0.05, min_value=0.0, max_value=0.2, key="zmin_" + title)
        zmin = np.percentile(data, quantile_min * 100)
    with subcol2:
        quantile_max = st.slider("Zmax", value=0.05, min_value=0.0, max_value=0.2, key="zmax_" + title)
        zmax = np.percentile(data, (1 - quantile_max) * 100)

    fig = px.imshow(data, color_continuous_scale=colormap, zmin=zmin, zmax=zmax, aspect="auto")
    fig.update_xaxes(tickvals=list(range(24)), ticktext=[str(i) for i in range(1, 25)])
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)


class FourierPage(PageModel):
    def render(self):
        st.title("Fourier analysis")
        marketindex: MarketIndex = st.session_state.get("marketindex")
        if marketindex is None:
            raise RenderWarning("MarketIndex not found, please load the data.")

        if not (dpt := st.session_state.get("dpt")):
            index_ticker = st.session_state.get("market_ticker")
            if index_ticker is None:
                raise RenderWarning("Please specify the index of the market index in the Import page.")

            dpt = DPT(data=marketindex.data, index_ticker=index_ticker)
            dpt.calculate_signals()
            st.session_state["dpt"] = dpt

        colormap = st.segmented_control(
            "Colormap", ["viridis", "plasma", "inferno", "magma", "cividis"], default="viridis"
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            set_graph(dpt.R.transpose(), "Power Spectral Density", column=col1, colormap=colormap)
        with col2:
            set_graph(dpt.coherence.transpose(), "Coherence with index", column=col2, colormap=colormap)


if __name__ == "__main__":
    FourierPage().run()
