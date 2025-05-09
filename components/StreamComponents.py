import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    xlabels: list | None = None,
    ylabels: list | None = None,
    controlable: bool = True,
) -> None:
    zmin, zmax = minmax_slider(data, title) if controlable else (None, None)
    x_len, y_len = data.shape
    xlabels = np.arange(1, x_len + 1) if xlabels is None else xlabels
    ylabels = np.arange(1, y_len + 1) if ylabels is None else ylabels
    data_t = data.transpose()
    fig = px.imshow(
        data_t, x=xlabels, y=ylabels, color_continuous_scale=colormap, zmin=zmin, zmax=zmax, aspect="auto", title=title
    )
    st.plotly_chart(fig, use_container_width=True)


def treemap(data: pl.DataFrame, values_label: str):
    data = data.unpivot(variable_name="cat", value_name=values_label)
    fig = px.treemap(data, path=["cat"], values=values_label, color=values_label, color_continuous_scale="RdYlGn")
    st.plotly_chart(fig)


def piechart(data: pl.DataFrame, values_label: str, pull: float = 0.05):
    data = data.unpivot(variable_name="cat", value_name=values_label)
    pulls = np.repeat(pull, len(data))
    fig = go.Figure(data=[go.Pie(labels=data["cat"], values=data[values_label], pull=pulls)])
    st.plotly_chart(fig)
