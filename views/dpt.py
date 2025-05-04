import numpy as np
import plotly.express as px
import streamlit as st

from models.Dpt import DPT
from models.Load import Indice


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


def graph(sp500: Indice):
    dpt = DPT(data=sp500.close, index_ticker="GSPC.INDX")
    colormap = st.selectbox(
        "Colormap", options=["viridis", "plasma", "inferno", "magma", "cividis"], label_visibility="collapsed"
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        set_graph(dpt.R, "Power Spectral Density", column=col1, colormap=colormap)
        set_graph(dpt.phase_shift, "Phase-shift", column=col1, colormap=colormap)
    with col2:
        set_graph(dpt.csd.real, "Cross-Spectral Density with SP500 Index", column=col2, colormap=colormap)
        set_graph(dpt.coherence, "Coherence", column=col2, colormap=colormap)


if __name__ == "__main__":
    sp500: Indice = st.session_state.get("sp500", None)
    if sp500:
        graph(sp500)
    else:
        st.warning("SP500 data not found, please load Data")
