import streamlit as st
import pandas as pd


def render_filters(df: pd.DataFrame):
    """
    Render sidebar filter controls and return the filtered DataFrame.

    Returns
    -------
    filtered_df : pd.DataFrame
    selected_sku : str or None  (None = bulk mode)
    """
    st.sidebar.header("Filters")

    # -- Store --
    store_list = ["All Stores"] + sorted(df["store_id"].unique().tolist())
    store = st.sidebar.selectbox("Store", store_list, key="filter_store")

    # -- Category --
    category_list = sorted(df["category"].unique().tolist())
    categories = st.sidebar.multiselect(
        "Category", category_list, default=category_list, key="filter_cat"
    )

    # -- Mode: bulk vs single SKU --
    sku_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Bulk (All SKUs)", "Single SKU"],
        key="filter_mode",
    )
    selected_sku = None
    if sku_mode == "Single SKU":
        sku_pool = df.copy()
        if store != "All Stores":
            sku_pool = sku_pool[sku_pool["store_id"] == store]
        if categories:
            sku_pool = sku_pool[sku_pool["category"].isin(categories)]
        sku_list = sorted(sku_pool["sku_id"].unique().tolist())
        if sku_list:
            selected_sku = st.sidebar.selectbox(
                "SKU", sku_list, key="filter_sku"
            )
        else:
            st.sidebar.warning("No SKUs available for current filters.")

    # -- Date Range --
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="filter_date",
    )

    # -- Apply filters --
    filtered = df.copy()

    if store != "All Stores":
        filtered = filtered[filtered["store_id"] == store]

    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    else:
        # Nothing selected → show warning, don't crash
        st.sidebar.warning("Select at least one category.")
        filtered = filtered.iloc[0:0]  # empty df

    if selected_sku:
        filtered = filtered[filtered["sku_id"] == selected_sku]

    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = filtered[
            (filtered["date"] >= start) & (filtered["date"] <= end)
        ]

    # -- Guard: empty result --
    if filtered.empty:
        st.sidebar.error("No data found for the selected filters.")
        st.stop()

    # -- Info --
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"**{len(filtered):,}** rows | "
        f"**{filtered['sku_id'].nunique()}** SKUs | "
        f"**{filtered['store_id'].nunique()}** stores"
    )

    return filtered, selected_sku
