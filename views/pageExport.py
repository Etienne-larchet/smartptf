from time import sleep
from typing import TYPE_CHECKING

import streamlit as st

from components.PageModels import PageModel, RenderWarning

if TYPE_CHECKING:
    from models.Load import MarketIndex


class ExportPage(PageModel):
    def __init__(self):
        st.set_page_config(page_title="Export", layout="wide")

    def render(self):
        st.title("Export")
        export_button = st.button("Save data", type="primary")

        marketindex: MarketIndex = st.session_state.get("marketindex")
        if marketindex is None and marketindex.data is None:
            raise RenderWarning("No data have been loaded yet. Please first load data via the Import page.")

        if export_button:
            marketindex.to_csv()
            success_placeholder = st.empty()  # Create a placeholder
            if marketindex.csv_data_path.exists():
                success_placeholder.success("✅ Validation successful!")
            else:
                success_placeholder.error("Error while saving data.")
            sleep(2)  # Display it for 3 seconds
            success_placeholder.empty()


if __name__ == "__main__":
    ExportPage().run()
