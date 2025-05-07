import streamlit as st

from components.MemorySession import MemorySession as mm
from components.PageModels import PageModel
from components.StreamComponents import heatmap


class FourierPage(PageModel):
    def render(self):
        st.title("Fourier Analysis")
        dpt = mm.dpt
        colormap = st.segmented_control(
            "Colormap", ["viridis", "plasma", "inferno", "magma", "cividis"], default="viridis"
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            heatmap(dpt.R, "Power Spectral Density", colormap=colormap)
            heatmap(dpt.coherence, "Coherence with index", colormap=colormap)
        with col2:
            heatmap(dpt.logR, "Log returns", colormap=colormap, x=dpt.get("logR")["Date"])


if __name__ == "__main__":
    FourierPage().run()
