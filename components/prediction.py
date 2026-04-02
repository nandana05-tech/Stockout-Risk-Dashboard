import streamlit as st
import pandas as pd

from utils.feature_engineering import run_feature_engineering
from utils.helpers import (
    get_risk_level,
    get_risk_emoji,
    get_days_left,
    get_action,
)


def render_prediction(filtered_df: pd.DataFrame, model, sku_id: str):
    """
    Deep-dive prediction for a single SKU.
    Shows 4 metric cards: probability, days left, risk level, action.
    Stores sku_df + analyzed flag in session_state for visualization.
    """
    st.subheader(f"SKU Analysis: {sku_id}")

    if st.button("Analyze SKU", key="btn_analyze"):
        sku_df = filtered_df[filtered_df["sku_id"] == sku_id].copy()

        if sku_df.empty:
            st.error("No data available for this SKU in the selected date range.")
            return

        with st.spinner("Running feature engineering & prediction..."):
            # Run FE on full SKU history (rolling features need history)
            X, df_fe = run_feature_engineering(sku_df)
            df_fe["risk_score"] = model.predict_proba(X)[:, 1]

        # Use latest row for current snapshot
        latest   = df_fe.iloc[-1]
        score    = float(latest["risk_score"])
        med_t    = st.session_state.get("medium_thresh", 0.5)
        high_t   = st.session_state.get("high_thresh",   0.8)
        risk     = get_risk_level(score, med_t, high_t)
        emoji      = get_risk_emoji(risk)
        rolling_7d = float(latest["rolling_7d_sales"])
        days_left  = get_days_left(float(latest["stock_on_hand"]), rolling_7d)
        action     = get_action(risk, days_left, float(latest["lead_time_days"]))

        # Display result cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Stockout Probability", f"{score:.1%}")
        col2.metric("Days Until Stockout",  f"{days_left:.1f} days")
        col3.metric("Risk Level",           f"{emoji} {risk}")
        col4.metric("Recommended Action",   action)

        # Store in session_state for visualization components
        st.session_state["sku_df"]   = df_fe   # FE-enriched version
        st.session_state["analyzed"] = True
        st.session_state["analyzed_sku"] = sku_id
