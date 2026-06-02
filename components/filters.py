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

        sku_df = (
            sku_pool[["sku_id", "sku_name"]]
            .drop_duplicates()
            .sort_values("sku_id")
            .reset_index(drop=True)
        )
        sku_df["sku_id"] = sku_df["sku_id"].astype(str)
        sku_df["sku_name"] = sku_df["sku_name"].astype(str)

        if not sku_df.empty:
            search_query = st.sidebar.text_input(
                "Cari SKU",
                key="sku_search",
                placeholder="Ketik nama atau ID SKU...",
            )

            if search_query:
                q = search_query.lower()
                mask = sku_df["sku_id"].str.lower().str.contains(
                    q, na=False, regex=False
                ) | sku_df["sku_name"].str.lower().str.contains(
                    q, na=False, regex=False
                )
                suggestions = sku_df[mask].head(10)

                # Bersihkan radio lama HANYA saat query berubah,
                # bukan setiap rerun — agar klik radio tidak ter-reset
                if st.session_state.get("_prev_sku_search") != search_query:
                    st.session_state.pop("filter_sku_radio", None)
                    st.session_state["_prev_sku_search"] = search_query

                if not suggestions.empty:
                    sku_options = (
                        suggestions["sku_id"] + " — " + suggestions["sku_name"]
                    ).tolist()
                    st.sidebar.caption(
                        f"{len(suggestions)} rekomendasi "
                        + ("(10 teratas)" if len(sku_df[mask]) > 10 else "")
                    )
                    selected_option = st.sidebar.radio(
                        "Pilih SKU:", sku_options, key="filter_sku_radio"
                    )
                    selected_sku = selected_option.split(" — ")[0]
                else:
                    st.sidebar.warning(
                        f"Tidak ada SKU yang cocok dengan '{search_query}'."
                    )
                    # Tandai agar main area bisa menampilkan pesan yang jelas
                    st.session_state["_sku_search_no_result"] = search_query
            else:
                sku_options = (
                    sku_df["sku_id"] + " — " + sku_df["sku_name"]
                ).tolist()
                selected_option = st.sidebar.selectbox(
                    "SKU", sku_options, key="filter_sku"
                )
                selected_sku = selected_option.split(" — ")[0]
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

    # Jika Single SKU mode tapi pencarian tidak menemukan hasil → hentikan, jangan lanjut ke ML
    if sku_mode == "Single SKU" and selected_sku is None:
        no_result_query = st.session_state.pop("_sku_search_no_result", None)
        if no_result_query:
            st.error(
                f"Tidak ada SKU yang cocok dengan **'{no_result_query}'**. "
                "Coba kata kunci lain atau kosongkan pencarian."
            )
        st.stop()

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
