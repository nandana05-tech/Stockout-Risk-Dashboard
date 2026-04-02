import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.feature_engineering import run_feature_engineering


def render_stock_chart(sku_df: pd.DataFrame):
    """
    Line chart: Stock On Hand vs Units Sold over time.
    sku_df should be the FE-enriched dataframe from session_state.
    """
    st.subheader("Stock vs Demand History")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sku_df["date"],
        y=sku_df["stock_on_hand"],
        name="Stock On Hand",
        line=dict(color="#4CAF50", width=2),
        fill="tozeroy",
        fillcolor="rgba(76,175,80,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=sku_df["date"],
        y=sku_df["units_sold"],
        name="Units Sold",
        line=dict(color="#FF5722", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=sku_df["date"],
        y=sku_df["rolling_7d_sales"],
        name="7-Day Avg Sales",
        line=dict(color="#FFC107", width=1.5, dash="dot"),
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Units",
        legend=dict(orientation="h", y=1.02),
        hovermode="x unified",
        margin=dict(t=40, b=40),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_risk_timeline(sku_df: pd.DataFrame, model, simulation_days: int = 7):
    """
    Forward-simulate 7 days and plot predicted stockout probability per day.

    Strategy:
    - Use last row as starting point
    - Decrease stock by avg_daily_sales each day
    - Concat simulated rows to history so rolling features stay accurate
    - Run FE on combined, take last `simulation_days` risk scores
    """
    st.subheader("7-Day Risk Forecast")

    last_row       = sku_df.iloc[-1].copy()
    avg_daily      = float(last_row["rolling_7d_sales"])
    last_date      = pd.to_datetime(last_row["date"])
    simulated_stock = float(last_row["stock_on_hand"])

    simulated_rows = []
    for d in range(1, simulation_days + 1):
        simulated_stock = max(0.0, simulated_stock - avg_daily)
        sim_row = last_row.copy()
        sim_row["stock_on_hand"] = simulated_stock
        sim_row["units_sold"]    = avg_daily
        sim_row["date"]          = last_date + pd.Timedelta(days=d)
        simulated_rows.append(sim_row)

    sim_df   = pd.DataFrame(simulated_rows).reset_index(drop=True)
    combined = pd.concat([sku_df, sim_df], ignore_index=True)

    with st.spinner("Computing risk forecast..."):
        X_combined, _ = run_feature_engineering(combined)
        all_scores     = model.predict_proba(X_combined)[:, 1]

    future_scores = all_scores[-simulation_days:]
    future_dates  = [
        (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(simulation_days)
    ]

    # Color points by risk level
    point_colors = []
    for s in future_scores:
        if s > 0.8:
            point_colors.append("#F44336")
        elif s > 0.5:
            point_colors.append("#FFC107")
        else:
            point_colors.append("#4CAF50")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_scores,
        mode="lines+markers",
        line=dict(color="#FF5722", width=2),
        marker=dict(color=point_colors, size=10, line=dict(width=1, color="white")),
        name="Predicted Risk",
        hovertemplate="Date: %{x}<br>Risk: %{y:.1%}<extra></extra>",
    ))

    # Threshold reference lines
    fig.add_hline(
        y=0.8, line_dash="dash", line_color="#F44336",
        annotation_text="High Threshold (80%)",
        annotation_position="top right",
    )
    fig.add_hline(
        y=0.5, line_dash="dash", line_color="#FFC107",
        annotation_text="Medium Threshold (50%)",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Stockout Probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        hovermode="x unified",
        margin=dict(t=40, b=40),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)
