from typing import Literal

import numpy as np
import streamlit as st

from components.MemorySession import MemorySession as mm
from components.PageModels import PageModel, RenderWarning
from components.StreamComponents import heatmap, piechart, treemap


def signals_sliders(
    type: Literal["Alpha", "Beta"],
    nb_signals: int,
    saved_signals: dict[int, float],
    minval: float = 0.01,
    maxval: float = 2.0,
    nb_cols: int = 3,
) -> dict[int, float]:
    params = dict(min_value=minval, max_value=maxval)
    topcol1, topcol2 = st.columns(2)
    with topcol2:
        details = mm.advancedcontrols_signals[type]
        details = st.toggle("Advanced controls", key=f"AdvancedControlsSignas{type}", value=details)
        mm.advancedcontrols_signals[type] = details

    with topcol1:
        mastersignal = st.slider(f"{type} Global", value=saved_signals[1], disabled=details, **params)
    if not details:
        for i in range(nb_signals):
            saved_signals[i + 1] = mastersignal
        return saved_signals
    subcols = st.columns(nb_cols)
    k = 1
    for _ in range(nb_signals // nb_cols + 1):
        for i in range(nb_cols):
            if k > nb_signals:
                break
            with subcols[i]:
                kth_value = st.slider(f"{type} signal {k}", value=saved_signals[k], **params)
                saved_signals[k] = kth_value
                k += 1
    return saved_signals


class SolverPage(PageModel):
    def render(self):
        st.title("Solver")

        # FORECASTS METHOD SELECTION
        forecasts = {
            "Moving Average": mm.forecast_ma,
            "AutoARIMA": mm.forecast_arima,
            "Exponential Smoothness": mm.forecast_es,
            "LSTM": mm.forecast_lstm,
        }
        loaded_forecasts = {k: v for k, v in forecasts.items() if v is not None}
        if not loaded_forecasts:
            raise RenderWarning("No forecasts loaded, please go to Forecasting Page.")

        forecast_labels = loaded_forecasts.keys()
        origin = st.pills("Forecasts origin", list(forecast_labels), default=list(forecast_labels)[0])

        # DISPLAY SELECTED FORECASTS IN TREEMAP
        treemap(loaded_forecasts[origin], values_label="logR")

        # DPT SETTINGS
        col1, col2, col3 = st.columns(3)
        with col1:
            mm.ptf_size = st.slider("Ptf size", min_value=2, max_value=100, value=mm.ptf_size)
        with col2:
            mm.min_invest = st.slider("Min. Investment", min_value=0.005, max_value=0.49, value=mm.min_invest)
        with col3:
            mm.max_invest = st.slider("Max. Investment", min_value=0.02, max_value=0.49, value=mm.max_invest)

        nb_signals = len(mm.dpt.R)
        col1, col2 = st.columns(2)
        with col1:
            mm.beta = signals_sliders("Beta", nb_signals=nb_signals, saved_signals=mm.beta, maxval=2.0, nb_cols=4)
        with col2:
            mm.alpha = signals_sliders("Alpha", nb_signals=nb_signals, saved_signals=mm.alpha, maxval=1.0, nb_cols=4)

        # DPT SOLVING
        if st.button("Solve", type="primary"):
            forecasts_dict = loaded_forecasts[origin].to_dicts()[0]
            solve_params = dict(
                mu=forecasts_dict, S=mm.ptf_size, C_betas=mm.beta, C_alphas=mm.alpha, L=mm.min_invest, M=mm.max_invest
            )
            mm.optptf = mm.dpt.solve(**solve_params)

        # DISPLAY RESULTS
        if mm.optptf:
            st.divider()
            st.header("Results")
            heatmap_params = dict(colormap="viridis", ylabels=mm.optptf.returns.columns, controlable=False)

            st.markdown(f"Expected Return (Monthly): **{mm.optptf.ptf_return:.1%}**")
            st.markdown(f"Expected Return (Annualy): **{(1 + mm.optptf.ptf_return) ** 12 - 1:.1%}**")
            col1, col2 = st.columns(2)
            with col1:
                piechart(mm.optptf.weights, "Weights")
                heatmap(mm.optptf.betas, "Betas", **heatmap_params)
                heatmap(mm.optptf.weighted_betas, "Weighted Betas", **heatmap_params)
                st.dataframe(mm.optptf.ptf_betas.to_pandas().set_index(np.arange(1, nb_signals + 1)))

            with col2:
                heatmap(mm.optptf.R, "PSD", **heatmap_params)
                heatmap(mm.optptf.alphas, "Alphas", **heatmap_params)
                heatmap(mm.optptf.weighted_alphas, "Weighted Alphas", **heatmap_params)

                st.dataframe(mm.optptf.ptf_alphas.to_pandas().set_index(np.arange(1, nb_signals + 1)))


if __name__ == "__main__":
    SolverPage().run()
