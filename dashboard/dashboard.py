import altair as alt
import pandas as pd
import streamlit as st

alt.data_transformers.disable_max_rows()
st.set_page_config(page_title="Dengue Prediction Dashboard", layout="wide")

st.title("Dengue Prediction in RS/BRA — Streamlit Dashboard")
st.caption("Based on the notebook: dengue.ipynb")

st.markdown("---")
st.markdown(
    "**Run locally:** `streamlit run dashboard.py`  \\\n    **Dependencies:** `pip install streamlit altair pandas`"
)