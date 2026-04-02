import streamlit as st

from utils.helpers import load_data, load_model
from components.filters import render_filters
from components.overview import render_overview
from components.prediction import render_prediction
from components.visualization import render_stock_chart, render_risk_timeline
from components.action_table import render_action_table

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FMCG Stockout Risk Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load shared resources
# ---------------------------------------------------------------------------
df    = load_data()
model = load_model()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
filtered_df, selected_sku = render_filters(df)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("FMCG Stockout Risk Dashboard")
st.caption(
    "Real-time stockout risk prediction powered by Calibrated LightGBM."
)

st.divider()

# ---------------------------------------------------------------------------
# Section 1: Overview KPIs (always shown)
# ---------------------------------------------------------------------------
latest_df = render_overview(filtered_df, model)

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Action Table (always shown)
# ---------------------------------------------------------------------------
render_action_table(latest_df)

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Single SKU Deep Dive (only in Single SKU mode)
# ---------------------------------------------------------------------------
if selected_sku:
    st.header(f"Deep Dive: {selected_sku}")

    render_prediction(filtered_df, model, selected_sku)

    if st.session_state.get("analyzed") and \
       st.session_state.get("analyzed_sku") == selected_sku:
        sku_df = st.session_state["sku_df"]

        col_left, col_right = st.columns(2)
        with col_left:
            render_stock_chart(sku_df)
        with col_right:
            render_risk_timeline(sku_df, model)
