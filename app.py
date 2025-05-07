import streamlit as st
from dotenv import load_dotenv

from config.logging_config import configure_logging

load_dotenv("config/.env")
configure_logging()


# --- PAGE SETUP ---
page_import = st.Page("views/pageImport.py", title="Import", default=True)
page_export = st.Page("views/pageExport.py", title="Export")
page_fourier = st.Page("views/pageFourierAnalysis.py", title="Fourier Analysis")
page_forecasting = st.Page("views/pageForecasting.py", title="Forecasting")
page_solver = st.Page("views/pageSolver.py", title="Solve")

pg = st.navigation({"Settings": [page_import, page_export], "DPT": [page_fourier, page_forecasting, page_solver]})
pg.run()
