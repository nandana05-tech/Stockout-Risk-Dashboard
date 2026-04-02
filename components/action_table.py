import streamlit as st
import pandas as pd
from datetime import timedelta

from utils.helpers import (
    get_risk_level,
    get_risk_emoji,
    get_days_left,
    get_action,
    suggest_order,
)


def _compute_future_dates(action_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add predicted future dates to the action table.

    predicted_stockout_date  = last data date + days_left
    restock_needed_by        = predicted_stockout_date - lead_time_days
                               (date you need to PLACE the order)
    is_urgent                = restock_needed_by <= today
    """
    # Use the 'date' column (last recorded date per SKU) as reference point
    if "date" not in action_df.columns:
        action_df["predicted_stockout_date"] = pd.NaT
        action_df["restock_needed_by"]       = pd.NaT
        return action_df

    ref_date = pd.to_datetime(action_df["date"])

    action_df["predicted_stockout_date"] = ref_date + action_df["days_left"].apply(
        lambda d: timedelta(days=float(d))
    )
    action_df["restock_needed_by"] = (
        action_df["predicted_stockout_date"]
        - action_df["lead_time_days"].apply(lambda d: timedelta(days=float(d)))
    )

    return action_df


def render_action_table(latest_df: pd.DataFrame):
    """
    Render ranked action table for all SKUs in latest_df.
    latest_df must already contain: risk_score, rolling_7d_sales,
    stock_on_hand, lead_time_days (produced by overview.py).
    """
    st.subheader("Action Table & Future Prediction")

    action_df = latest_df.copy()

    # Read dynamic thresholds set by overview.py
    medium_thresh = st.session_state.get("medium_thresh", 0.5)
    high_thresh   = st.session_state.get("high_thresh",   0.8)

    # Fill NaN in columns used for arithmetic before apply()
    action_df["rolling_7d_sales"] = action_df["rolling_7d_sales"].fillna(0.0)
    action_df["lead_time_days"]   = action_df["lead_time_days"].fillna(0.0)
    action_df["stock_on_hand"]    = action_df["stock_on_hand"].fillna(0.0)

    # --- Core derived columns ---
    action_df["days_left"] = action_df.apply(
        lambda r: get_days_left(r["stock_on_hand"], r["rolling_7d_sales"]),
        axis=1,
    )
    action_df["risk_level"] = action_df["risk_score"].apply(
        lambda s: get_risk_level(s, medium_thresh, high_thresh)
    )
    action_df["suggested_order"] = action_df.apply(
        lambda r: suggest_order(r["rolling_7d_sales"], r["lead_time_days"]),
        axis=1,
    )
    action_df["action"] = action_df.apply(
        lambda r: get_action(r["risk_level"], r["days_left"], r["lead_time_days"]),
        axis=1,
    )
    action_df["risk_display"] = action_df["risk_level"].apply(
        lambda r: f"{get_risk_emoji(r)} {r}"
    )

    # --- Future date prediction ---
    action_df = _compute_future_dates(action_df)

    # --- Filter controls ---
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    show_high_only = col_f1.checkbox("High Risk Only", key="tbl_high_only")
    show_action_filter = col_f2.selectbox(
        "Action Filter",
        ["All", "Restock NOW", "Order Soon", "Monitor", "Safe"],
        key="tbl_action_filter",
    )

    display_df = action_df.copy()
    if show_high_only:
        display_df = display_df[display_df["risk_level"] == "High"]
    if show_action_filter != "All":
        display_df = display_df[display_df["action"] == show_action_filter]

    display_df = display_df.sort_values("risk_score", ascending=False)

    # --- Select display columns ---
    cols_to_show = [
        "sku_id", "sku_name", "store_id", "category",
        "risk_display", "risk_score", "days_left",
        "predicted_stockout_date", "restock_needed_by",
        "stock_on_hand", "rolling_7d_sales",
        "lead_time_days", "suggested_order", "action",
    ]
    cols_to_show = [c for c in cols_to_show if c in display_df.columns]
    display_df = display_df[cols_to_show]

    st.caption(f"Showing **{len(display_df):,}** SKU-Store pairs")

    # --- Render table ---
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score",
                min_value=0,
                max_value=1,
                format="%.4f",
            ),
            "days_left": st.column_config.NumberColumn(
                "Days Left",
                format="%.1f d",
            ),
            "predicted_stockout_date": st.column_config.DateColumn(
                "Predicted Stockout Date",
                format="YYYY-MM-DD",
            ),
            "restock_needed_by": st.column_config.DateColumn(
                "Restock Needed By",
                format="YYYY-MM-DD",
            ),
            "rolling_7d_sales": st.column_config.NumberColumn(
                "Avg Daily Sales (7d)",
                format="%.1f",
            ),
            "suggested_order": st.column_config.NumberColumn(
                "Suggested Order Qty",
                format="%d",
            ),
            "risk_display": st.column_config.TextColumn("Risk Level"),
        },
    )

    # --- CSV Export ---
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export to CSV",
        data=csv_bytes,
        file_name="stockout_action_table.csv",
        mime="text/csv",
        key="download_action_table",
    )
