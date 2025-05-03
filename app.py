import streamlit as st
from dotenv import load_dotenv

from config.logging_config import configure_logging

load_dotenv("config/.env")
configure_logging()


# --- PAGE SETUP ---
load_page = st.Page("views/load.py", title="Load", default=True)

dpt_page = st.Page("views/dpt.py", title="DPT")


pg = st.navigation([load_page, dpt_page])
pg.run()
