import numpy as np
import plotly.express as px
import streamlit as st

from models.dpt import DPT
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


# fig, axes = plt.subplots(4, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [1, 1.5]})
# x_full = np.tile(np.arange(R_np.shape[1]), R_np.shape[0])
# x_assets = np.tile(np.arange(R_np.shape[1]), R_np.shape[0]-1)

# # Power Spectral Density
# axes[0,0].bar(x_full, R_np.ravel(), alpha=0.01)
# axes[0,0].set_title('Welch Fourier Transform of returns')
# axes[0,0].set_xlabel('kth')
# axes[0,0].set_ylabel('Magnitude')
# axes[0,0].set_xlim(0)

# cax0 = axes[0,1].imshow(R_np, aspect='auto', cmap='viridis')
# axes[0,1].set_title("Heatmap")
# axes[0,1].set_xlabel("kth")
# axes[0,1].set_ylabel("Assets")
# fig.colorbar(cax0, label='SPD', ax=axes[0,1])

# #Cross-Spectral Density
# axes[1,0].bar(x_assets, Cxy.ravel(), alpha=0.01)
# axes[1,0].set_title('Cross-Spectral Density with Index')
# axes[1,0].set_xlabel('kth')
# axes[1,0].set_ylabel('Magnitude')
# axes[1,0].set_xlim(0)

# cax1 = axes[1,1].imshow(Cxy.real, aspect='auto', cmap='viridis')
# axes[1,1].set_xlabel("kth")
# axes[1,1].set_ylabel("Assets")
# fig.colorbar(cax1, label='SPD', ax=axes[1,1])

# # Coherence
# axes[2,0].bar(x_assets, Ccoh.ravel(), alpha=0.01)
# axes[2,0].set_title('Coherence with Index')
# axes[2,0].set_xlabel('kth')
# axes[2,0].set_ylabel('Magnitude')
# axes[2,0].set_xlim(0)

# cax2 = axes[2,1].imshow(Ccoh.real, aspect='auto', cmap='viridis')
# axes[2,1].set_xlabel("kth")
# axes[2,1].set_ylabel("Assets")
# fig.colorbar(cax2, label='SPD', ax=axes[2,1])

# # Phase shift
# axes[3,0].bar(x_assets, phase.ravel(), alpha=0.01)
# axes[3,0].set_title('Phase Shift with Index')
# axes[3,0].set_xlabel('kth')
# axes[3,0].set_ylabel('Magnitude')
# axes[3,0].set_xlim(0)

# cax2 = axes[3,1].imshow(phase.real, aspect='auto', cmap='viridis')
# axes[3,1].set_xlabel("kth")
# axes[3,1].set_ylabel("Assets")
# fig.colorbar(cax2, label='SPD', ax=axes[3,1])

# plt.tight_layout(pad=0.5, h_pad=2.0)
# plt.show()
