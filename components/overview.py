import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from utils.feature_engineering import run_feature_engineering
from utils.helpers import (
    get_risk_level,
    get_risk_emoji,
    get_latest_per_sku,
    compute_risk_thresholds,
    get_days_left,
)


def render_overview(filtered_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Run batch prediction on the latest record per SKU-Store combination
    and display KPI summary cards.

    Returns
    -------
    latest_df : pd.DataFrame with risk_score, risk_level, and FE columns
    """
    with st.spinner("Running batch risk analysis..."):
        n_sku_est = filtered_df.groupby(["sku_id", "store_id"], observed=True).ngroups
        if n_sku_est > 5000:
            st.warning(
                f"Large batch: ~{n_sku_est:,} SKU-Store pairs. "
                "Consider narrowing filters for faster results."
            )

        # 1. Run FE on full filtered history (rolling features need history)
        X_full, df_fe_full = run_feature_engineering(filtered_df)
        df_fe_full["risk_score"] = model.predict_proba(X_full)[:, 1]

        # 2. Keep only latest date per (sku_id, store_id)
        df_fe_full["date"] = pd.to_datetime(df_fe_full["date"])
        latest_df = (
            df_fe_full.sort_values("date")
            .groupby(["sku_id", "store_id"], as_index=False, observed=True)
            .last()
            .reset_index(drop=True)
        )

        # 3. Compute dynamic thresholds from score distribution
        medium_thresh, high_thresh = compute_risk_thresholds(latest_df["risk_score"])
        st.session_state["medium_thresh"] = medium_thresh
        st.session_state["high_thresh"]   = high_thresh

        latest_df["risk_level"] = latest_df["risk_score"].apply(
            lambda s: get_risk_level(s, medium_thresh, high_thresh)
        )

    # 4. Compute KPIs
    high_count = (latest_df["risk_level"] == "High").sum()
    med_count  = (latest_df["risk_level"] == "Medium").sum()
    low_count  = (latest_df["risk_level"] == "Low").sum()
    total_skus = len(latest_df)
    avg_score  = latest_df["risk_score"].mean()

    # 5. Render KPI cards
    st.subheader("Risk Overview — Current State")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total SKUs", f"{total_skus:,}")
    col2.metric("🔴 High Risk", f"{high_count:,}",
                delta=f"{high_count/total_skus:.1%}" if total_skus else "0%",
                delta_color="inverse")
    col3.metric("🟡 Medium Risk", f"{med_count:,}",
                delta=f"{med_count/total_skus:.1%}" if total_skus else "0%",
                delta_color="off")
    col4.metric("🟢 Low Risk", f"{low_count:,}")
    col5.metric("Avg Risk Score", f"{avg_score:.4f}")

    with st.expander("Risk Threshold Info"):
        st.caption(
            f"Thresholds computed from batch score distribution. "
            f"**High** ≥ {high_thresh:.4f} (top 5%), "
            f"**Medium** ≥ {medium_thresh:.4f} (top 25%)."
        )

    # =========================================================================
    # 7. Future Risk Projection — date range picker
    # =========================================================================
    st.subheader("Future Risk Projection")
    st.caption(
        "Pilih rentang tanggal proyeksi. Referensi = tanggal data terakhir per SKU. "
        "Simulasi menggunakan rata-rata permintaan 7 hari untuk memproyeksikan "
        "kapan stok setiap SKU akan habis."
    )

    # Build per-SKU stockout forecast (no model re-run needed)
    sim_df = latest_df.copy()
    sim_df["rolling_7d_sales"] = sim_df["rolling_7d_sales"].fillna(0.0)
    sim_df["stock_on_hand"]    = sim_df["stock_on_hand"].fillna(0.0)
    sim_df["lead_time_days"]   = sim_df["lead_time_days"].fillna(7.0)
    sim_df["ref_date"]         = pd.to_datetime(sim_df["date"])

    sim_df["days_left_sim"] = sim_df.apply(
        lambda r: get_days_left(r["stock_on_hand"], r["rolling_7d_sales"]),
        axis=1,
    )
    sim_df["predicted_stockout_date"] = sim_df["ref_date"] + pd.to_timedelta(
        sim_df["days_left_sim"], unit="D"
    )
    sim_df["restock_needed_by"] = sim_df["predicted_stockout_date"] - pd.to_timedelta(
        sim_df["lead_time_days"], unit="D"
    )

    # Date range controls
    import datetime as _dt
    ref_min       = sim_df["ref_date"].max().date()
    default_start = ref_min
    default_end   = ref_min + _dt.timedelta(days=30)

    col_d1, col_d2 = st.columns(2)
    proj_start = col_d1.date_input(
        "Dari Tanggal", value=default_start, key="proj_start",
        help="Mulai rentang proyeksi",
    )
    proj_end = col_d2.date_input(
        "Sampai Tanggal", value=default_end, key="proj_end",
        help="Akhir rentang proyeksi — bisa tahun depan atau lebih",
    )

    if proj_end <= proj_start:
        st.warning("Tanggal akhir harus setelah tanggal awal.")
        return latest_df

    proj_start_ts = pd.Timestamp(proj_start)
    proj_end_ts   = pd.Timestamp(proj_end)

    # Categorize each SKU
    already_out     = sim_df["predicted_stockout_date"] < proj_start_ts
    in_window       = (
        (sim_df["predicted_stockout_date"] >= proj_start_ts) &
        (sim_df["predicted_stockout_date"] <= proj_end_ts)
    )
    order_in_window = (
        (sim_df["restock_needed_by"] >= proj_start_ts) &
        (sim_df["restock_needed_by"] <= proj_end_ts)
    )
    still_safe      = sim_df["predicted_stockout_date"] > proj_end_ts

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔴 Stockout Sebelum Window", f"{already_out.sum():,}",
              help=f"Stok habis sebelum {proj_start}")
    k2.metric("⚠️ Stockout Dalam Periode", f"{in_window.sum():,}",
              help=f"Stok habis antara {proj_start} s/d {proj_end}")
    k3.metric("📦 Harus Order Dalam Periode", f"{order_in_window.sum():,}",
              help="Order harus ditempatkan dalam rentang ini agar tidak stockout")
    k4.metric("🟢 Aman Sampai Akhir Period", f"{still_safe.sum():,}",
              help=f"Stok masih ada setelah {proj_end}")

    # -------------------------------------------------------------------------
    # Cumulative stockout chart
    # Chart starts from the REFERENCE DATE so cumulative begins at 0.
    # Window boundaries shown as vertical marker traces (avoids add_vline bug
    # with string x-axis in Plotly 6.x).
    # -------------------------------------------------------------------------
    import plotly.graph_objects as go

    chart_start      = sim_df["ref_date"].max()
    chart_end        = proj_end_ts
    total_chart_days = max(1, (chart_end - chart_start).days)
    num_pts          = max(2, total_chart_days // 7 + 1)

    chart_dates_dt, so_curve, ord_curve = [], [], []
    for w in range(num_pts):
        pt = chart_start + pd.Timedelta(days=7 * w)
        chart_dates_dt.append(pt.to_pydatetime())
        so_curve.append(int((sim_df["predicted_stockout_date"] <= pt).sum()))
        ord_curve.append(int((sim_df["restock_needed_by"] <= pt).sum()))
    # Ensure last point exactly at chart_end
    chart_dates_dt.append(chart_end.to_pydatetime())
    so_curve.append(int((sim_df["predicted_stockout_date"] <= chart_end).sum()))
    ord_curve.append(int((sim_df["restock_needed_by"] <= chart_end).sum()))

    max_y = max(max(so_curve), max(ord_curve), total_skus, 1)

    fig = go.Figure()

    # Main curves
    fig.add_trace(go.Scatter(
        x=chart_dates_dt, y=so_curve,
        mode="lines+markers", name="Cumulative Stockouts",
        line=dict(color="#F44336", width=2), marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(244,67,54,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=chart_dates_dt, y=ord_curve,
        mode="lines+markers", name="Cumulative Orders Needed",
        line=dict(color="#FF9800", width=2, dash="dot"), marker=dict(size=5),
    ))

    # Window start — vertical line via add_shape (works on date axis)
    fig.add_shape(
        type="line",
        xref="x", yref="y",
        x0=proj_start_ts.to_pydatetime(), x1=proj_start_ts.to_pydatetime(),
        y0=0, y1=max_y,
        line=dict(color="#2196F3", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=proj_start_ts.to_pydatetime(), y=max_y,
        text=f"Window Start<br>{proj_start}",
        showarrow=False, font=dict(color="#2196F3", size=10),
        xanchor="right", yanchor="top",
    )

    # Window end — vertical line
    fig.add_shape(
        type="line",
        xref="x", yref="y",
        x0=proj_end_ts.to_pydatetime(), x1=proj_end_ts.to_pydatetime(),
        y0=0, y1=max_y,
        line=dict(color="#9C27B0", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=proj_end_ts.to_pydatetime(), y=max_y,
        text=f"Window End<br>{proj_end}",
        showarrow=False, font=dict(color="#9C27B0", size=10),
        xanchor="left", yanchor="top",
    )

    # Total SKUs reference
    fig.add_hline(
        y=total_skus, line_dash="dash", line_color="#9E9E9E",
        annotation_text="Total SKUs", annotation_position="top right",
    )

    fig.update_layout(
        title=(
            f"Projected Cumulative Stockouts  |  "
            f"Ref: {chart_start.date()} → {proj_end}"
        ),
        xaxis=dict(title="Date", type="date"),
        yaxis=dict(title="Cumulative SKU Count"),
        legend=dict(orientation="h", y=1.02),
        hovermode="x unified",
        height=370,
        margin=dict(t=70, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Garis biru = awal window ({proj_start})  |  "
        f"Garis ungu = akhir window ({proj_end})  |  "
        f"Kurva dimulai dari tanggal referensi ({chart_start.date()}) "
        f"sehingga cumulative = 0 di awal."
    )

    # At-risk SKU detail table
    at_risk_in_window = sim_df[in_window | already_out].copy()
    if not at_risk_in_window.empty:
        with st.expander(
            f"SKU yang stockout dalam periode ini ({len(at_risk_in_window):,} SKUs)"
        ):
            show_cols = [c for c in [
                "sku_id", "sku_name", "store_id", "category",
                "stock_on_hand", "rolling_7d_sales",
                "predicted_stockout_date", "restock_needed_by",
                "days_left_sim",
            ] if c in at_risk_in_window.columns]
            st.dataframe(
                at_risk_in_window[show_cols]
                .sort_values("predicted_stockout_date")
                .reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "predicted_stockout_date": st.column_config.DateColumn(
                        "Predicted Stockout", format="YYYY-MM-DD"),
                    "restock_needed_by": st.column_config.DateColumn(
                        "Restock Needed By", format="YYYY-MM-DD"),
                    "days_left_sim": st.column_config.NumberColumn(
                        "Days Left", format="%.1f d"),
                },
            )

    return latest_df
