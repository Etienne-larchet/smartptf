import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st


def minmax_slider(
    data: np.ndarray | pl.DataFrame, id: str, min_val: float = 0.0, max_val: float = 0.25
) -> tuple[float, float]:
    col1, col2, _ = st.columns(3)
    with col1:
        quantile_min = st.slider("Zmin", value=0.05, min_value=min_val, max_value=max_val, key=f"zmin_{id}")
        zmin = np.percentile(data, quantile_min * 100)
    with col2:
        quantile_max = st.slider("Zmax", value=0.05, min_value=min_val, max_value=max_val, key=f"zmax_{id}")
        zmax = np.percentile(data, (1 - quantile_max) * 100)
    return float(zmin), float(zmax)


def heatmap(
    data: np.ndarray | pl.DataFrame,
    title: str,
    colormap: str | None = None,
    x: list | None = None,
    controlable: bool = True,
) -> None:
    zmin, zmax = minmax_slider(data, title) if controlable else (None, None)
    x = np.arange(1, len(data) + 1) if x is None else x
    data_t = data.transpose()
    fig = px.imshow(data_t, x=x, color_continuous_scale=colormap, zmin=zmin, zmax=zmax, aspect="auto", title=title)
    st.plotly_chart(fig, use_container_width=True)
