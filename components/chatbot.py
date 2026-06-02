import streamlit as st
import pandas as pd

from components.overview import compute_batch_prediction, _df_fingerprint
from utils.helpers import compute_risk_thresholds, get_risk_level


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _get_latest_with_risk(filtered_df: pd.DataFrame, model) -> pd.DataFrame:
    """Compute latest_df with risk_level — always uses the same thresholds as Dashboard."""
    latest_df = compute_batch_prediction(filtered_df, model)
    # Selalu hitung threshold dari data aktual yang sama, bukan dari session_state
    # yang mungkin ditulis oleh Dashboard dengan filter berbeda.
    medium_thresh, high_thresh = compute_risk_thresholds(latest_df["risk_score"])
    latest_df = latest_df.copy()
    latest_df["risk_level"] = latest_df["risk_score"].apply(
        lambda s: get_risk_level(s, medium_thresh, high_thresh)
    )
    return latest_df


_MONTH_MAP = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}


def _build_context(latest_df: pd.DataFrame, filtered_df: pd.DataFrame) -> str:
    total_skus   = latest_df["sku_id"].nunique()
    total_stores = latest_df["store_id"].nunique()
    total_rows   = len(filtered_df)

    risk_counts = (
        latest_df["risk_level"].value_counts().to_dict()
        if "risk_level" in latest_df.columns else {}
    )
    lines = [
        "=== RINGKASAN DASHBOARD ===",
        f"Total SKU: {total_skus} | Total Toko: {total_stores} | Total Baris Data: {total_rows:,}",
        f"Risiko Tinggi: {risk_counts.get('High', 0)} SKU | "
        f"Risiko Menengah: {risk_counts.get('Medium', 0)} SKU | "
        f"Risiko Rendah: {risk_counts.get('Low', 0)} SKU",
    ]

    # Breakdown risiko per kategori (dari model)
    if "category" in latest_df.columns and "risk_level" in latest_df.columns:
        cat_summary = (
            latest_df.groupby("category", observed=True)["risk_level"]
            .value_counts().unstack(fill_value=0)
            .reindex(columns=["High", "Medium", "Low"], fill_value=0)
        )
        lines.append("\nRisiko per Kategori (prediksi model):")
        lines.append(cat_summary.to_string())

    # Top 10 SKU berisiko tertinggi
    if "risk_score" in latest_df.columns:
        cols = ["sku_id", "store_id", "category", "risk_level", "risk_score", "stock_on_hand"]
        cols += [c for c in ["sku_name", "rolling_7d_sales"] if c in latest_df.columns]
        lines.append("\n10 SKU Risiko Tertinggi Saat Ini:")
        lines.append(latest_df.nlargest(10, "risk_score")[cols].to_string(index=False))

    # ── Analisis historis dari data asli ────────────────────────────────────
    if "stock_out_flag" in filtered_df.columns:
        grp_cols = ["sku_id"] + (["sku_name"] if "sku_name" in filtered_df.columns else [])

        # Total hari stockout per SKU (semua periode) — tanpa copy()
        sku_total = (
            filtered_df.groupby(grp_cols, observed=True)["stock_out_flag"]
            .sum().reset_index()
            .rename(columns={"stock_out_flag": "total_stockout_days"})
            .nlargest(20, "total_stockout_days")
        )
        lines.append("\nTop 20 SKU — Total Hari Stock Out (semua periode):")
        lines.append(sku_total.to_string(index=False))

        # Top 5 SKU stockout per bulan-tahun — vectorized, tanpa iterrows()
        if "month" in filtered_df.columns and "year" in filtered_df.columns:
            grp_ym = ["year", "month"] + grp_cols
            ym_sku = (
                filtered_df.groupby(grp_ym, observed=True)["stock_out_flag"]
                .sum().reset_index()
                .rename(columns={"stock_out_flag": "stockout_days"})
            )
            ym_sku = ym_sku[ym_sku["stockout_days"] > 0].copy()
            has_name = "sku_name" in ym_sku.columns

            month_lines = [
                "\nTop 5 SKU Stock Out per Bulan per Tahun "
                "(stockout_days = hari individu dalam bulan+tahun tersebut, maks ~31):"
            ]
            for m in sorted(ym_sku["month"].unique()):
                month_name = _MONTH_MAP.get(int(m), str(m))
                month_lines.append(f"  {month_name}:")
                for yr in sorted(ym_sku["year"].unique()):
                    top5 = ym_sku[
                        (ym_sku["month"] == m) & (ym_sku["year"] == yr)
                    ].nlargest(5, "stockout_days")
                    if top5.empty:
                        continue
                    month_lines.append(f"    Tahun {yr}:")
                    # Vectorized string building — tanpa iterrows()
                    if has_name:
                        entries = (
                            "      - " + top5["sku_id"].astype(str)
                            + " (" + top5["sku_name"].astype(str) + "): "
                            + top5["stockout_days"].astype(int).astype(str) + " hari"
                        ).tolist()
                    else:
                        entries = (
                            "      - " + top5["sku_id"].astype(str) + ": "
                            + top5["stockout_days"].astype(int).astype(str) + " hari"
                        ).tolist()
                    month_lines.extend(entries)
            lines.extend(month_lines)

            # Ringkasan tingkat stockout per bulan
            monthly_rate = (
                filtered_df.groupby("month")["stock_out_flag"]
                .agg(stockout_days="sum", total_records="count")
                .assign(pct=lambda x: (x["stockout_days"] / x["total_records"] * 100).round(1))
            )
            monthly_rate.index = monthly_rate.index.map(
                lambda m: _MONTH_MAP.get(int(m), str(m))
            )
            lines.append("\nTingkat Stock Out per Bulan — semua tahun digabung (% dari total hari):")
            lines.append(monthly_rate.to_string())

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenAI streaming generator
# ---------------------------------------------------------------------------

def _stream_openai(api_key: str, history: list, context: str):
    try:
        from openai import OpenAI
    except ImportError:
        yield "Library `openai` belum terinstall. Jalankan: `pip install openai`"
        return

    client = OpenAI(api_key=api_key)

    system_prompt = (
        "Kamu adalah AI Assistant untuk dashboard manajemen inventori FMCG "
        "(Fast-Moving Consumer Goods). Tugasmu membantu pengguna memahami "
        "risiko stockout, menganalisis data stok, dan memberikan rekomendasi "
        "tindakan berdasarkan data yang tersedia.\n\n"
        "PENTING — cara membaca data stockout:\n"
        "- Data mencakup 3 tahun historis (2021, 2022, 2023).\n"
        "- 'total_stockout_days' adalah jumlah hari stock out di SELURUH PERIODE, "
        "bukan dalam satu bulan. Contoh: 45 hari di Januari = rata-rata 15 hari/tahun "
        "selama 3 tahun (Januari 2021 + 2022 + 2023).\n"
        "- 'risk_score' berasal dari prediksi model LightGBM, bukan perhitungan manual.\n"
        "- Selalu jelaskan konteks tahun saat menjawab pertanyaan tentang frekuensi stockout.\n\n"
        "Gunakan bahasa Indonesia yang jelas dan ringkas. Jika relevan, "
        "gunakan bullet point atau tabel markdown agar mudah dibaca. "
        "Jawab HANYA berdasarkan data yang diberikan di bawah ini — jangan mengarang.\n\n"
        + context
    )

    messages = [{"role": "system", "content": system_prompt}] + [
        m for m in history if m["role"] != "system"
    ]

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=2048,
            temperature=0.4,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"Terjadi kesalahan: {e}"


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

_SUGGESTIONS = [
    "SKU mana yang harus segera direstock minggu ini?",
    "Berapa banyak SKU berisiko tinggi per kategori?",
    "Toko mana yang paling banyak SKU berbahaya?",
    "Rekomendasikan prioritas tindakan untuk hari ini.",
]


def render_chatbot(filtered_df: pd.DataFrame, model):
    # ── API Key form (hanya tampil jika belum ada key) ──────────────────────
    if "openai_key" not in st.session_state:
        st.markdown("### Masukkan OpenAI API Key")
        st.caption(
            "Key hanya tersimpan selama sesi ini dan hanya dikirim ke OpenAI — "
            "tidak disimpan di server manapun."
        )

        col_input, col_btn = st.columns([5, 1])
        with col_input:
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="sk-...",
                label_visibility="collapsed",
                key="openai_key_input",
            )
        with col_btn:
            activate = st.button("Aktifkan", type="primary", use_container_width=True)

        st.caption("Key akan diverifikasi saat pesan pertama dikirim.")

        if activate:
            if api_key and api_key.startswith("sk-"):
                st.session_state["openai_key"] = api_key
            else:
                st.error("API key tidak valid. Harus diawali dengan 'sk-'.")

        if "openai_key" not in st.session_state:
            return  # belum aktif, hentikan render

    # ── Header ───────────────────────────────────────────────────────────────
    col_title, col_clear = st.columns([6, 1])
    with col_title:
        st.caption("Model: gpt-4o-mini · Konteks diperbarui otomatis sesuai filter aktif")
    with col_clear:
        if st.button("Reset", use_container_width=True):
            st.session_state["chat_messages"] = []

    # ── Inisialisasi riwayat ─────────────────────────────────────────────────
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # ── Deteksi perubahan filter — reset cache konteks & toast flag ──────────
    current_fp = _df_fingerprint(filtered_df)
    if st.session_state.get("_filter_fp") != current_fp:
        st.session_state["_filter_fp"]      = current_fp
        st.session_state["_context_cached"] = False
        st.session_state.pop("_cached_context", None)

    # ── Chat input dipanggil lebih awal — Streamlit selalu render di bawah ──
    prompt = st.chat_input("Tanya sesuatu tentang data stok Anda...")
    if prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})

    # ── Area pesan (scrollable, pesan input selalu di bawah area ini) ────────
    with st.container(height=540, border=False):
        messages = st.session_state["chat_messages"]

        # Welcome message saat kosong
        if not messages:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(
                    "Halo! Saya siap membantu Anda menganalisis data inventori.\n\n"
                    "Berikut beberapa hal yang bisa Anda tanyakan:"
                )
                for s in _SUGGESTIONS:
                    st.markdown(f"- *{s}*")

        # Render semua riwayat pesan
        for msg in messages:
            if msg["role"] == "user":
                _, col = st.columns([1, 3])
                with col:
                    with st.chat_message("user"):
                        st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg["content"])

        # Jika pesan terakhir dari user → generate respons AI di dalam container
        if messages and messages[-1]["role"] == "user":
            if not st.session_state.get("_context_cached", False):
                st.toast("Data cukup besar, mohon tunggu sebentar ya...", icon="📊")
            st.toast("Sedang memproses pertanyaan Anda...", icon="⏳")

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Memuat konteks data..."):
                    latest_df = _get_latest_with_risk(filtered_df, model)
                    # Cache konteks agar groupby berat tidak diulang setiap pesan
                    if "_cached_context" not in st.session_state:
                        st.session_state["_cached_context"] = _build_context(
                            latest_df, filtered_df
                        )
                    st.session_state["_context_cached"] = True

                context = st.session_state["_cached_context"]
                response = st.write_stream(
                    _stream_openai(
                        api_key=st.session_state["openai_key"],
                        history=messages,
                        context=context,
                    )
                )

            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": response}
            )
