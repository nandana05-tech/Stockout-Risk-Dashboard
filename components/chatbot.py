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
    has_name  = "sku_name" in filtered_df.columns
    has_store = "store_id" in filtered_df.columns
    has_cat   = "category" in filtered_df.columns
    has_year  = "year" in filtered_df.columns
    has_month = "month" in filtered_df.columns
    has_flag  = "stock_out_flag" in filtered_df.columns

    sku_name_cols = (["sku_name"] if has_name else [])
    grp_sku       = ["sku_id"] + sku_name_cols

    total_skus   = latest_df["sku_id"].nunique()
    total_stores = latest_df["store_id"].nunique() if has_store else "-"
    total_rows   = len(filtered_df)

    risk_counts = (
        latest_df["risk_level"].value_counts().to_dict()
        if "risk_level" in latest_df.columns else {}
    )

    lines = [
        "=== RINGKASAN DASHBOARD ===",
        f"Total SKU: {total_skus} | Total Toko: {total_stores} | Total Baris Data: {total_rows:,}",
        f"Risiko Tinggi: {risk_counts.get('High', 0)} pasang SKU-Toko | "
        f"Risiko Menengah: {risk_counts.get('Medium', 0)} | "
        f"Risiko Rendah: {risk_counts.get('Low', 0)}",
    ]

    # ── 1. Risiko per kategori ───────────────────────────────────────────────
    if has_cat and "risk_level" in latest_df.columns:
        cat_risk = (
            latest_df.groupby("category", observed=True)["risk_level"]
            .value_counts().unstack(fill_value=0)
            .reindex(columns=["High", "Medium", "Low"], fill_value=0)
        )
        cat_risk["Total_SKU_store"] = cat_risk.sum(axis=1)
        lines.append("\nRisiko per Kategori (jumlah pasang SKU-Toko):")
        lines.append(cat_risk.to_string())

    # ── 2. Top 15 SKU-Toko risiko tertinggi (tanpa risk_score) ──────────────
    if "risk_score" in latest_df.columns:
        cols = ["sku_id", "store_id", "risk_level", "stock_on_hand"]
        cols += [c for c in ["sku_name", "category", "rolling_7d_sales"] if c in latest_df.columns]
        lines.append("\n15 SKU-Toko Risiko Tertinggi Saat Ini:")
        lines.append(latest_df.nlargest(15, "risk_score")[cols].to_string(index=False))

    # ── 3. Kondisi terkini semua SKU-Toko High+Medium (stok & penjualan) ────
    if "risk_level" in latest_df.columns and has_store:
        hm = latest_df[latest_df["risk_level"].isin(["High", "Medium"])].copy()
        cols_hm = ["sku_id", "store_id", "risk_level", "stock_on_hand"]
        cols_hm += [c for c in ["sku_name", "category", "rolling_7d_sales"] if c in latest_df.columns]
        lines.append("\nKondisi Terkini SKU-Toko Risiko Tinggi & Menengah (stok saat ini):")
        lines.append(hm[cols_hm].sort_values(["sku_id", "store_id"]).to_string(index=False))

    # ── 4. Total stockout per kategori ───────────────────────────────────────
    if has_flag and has_cat:
        cat_so = (
            filtered_df.groupby("category", observed=True)["stock_out_flag"]
            .agg(total_stockout_days="sum", total_records="count")
            .assign(pct_hari=lambda x: (x["total_stockout_days"] / x["total_records"] * 100).round(1))
            .sort_values("total_stockout_days", ascending=False)
        )
        lines.append("\nTotal Hari Stockout per Kategori (semua periode):")
        lines.append(cat_so.to_string())

    # ── 5. Total stockout per SKU (semua periode) ────────────────────────────
    if has_flag:
        sku_total = (
            filtered_df.groupby(grp_sku, observed=True)["stock_out_flag"]
            .sum().reset_index()
            .rename(columns={"stock_out_flag": "total_stockout_days"})
            .sort_values("total_stockout_days", ascending=False)
        )
        lines.append("\nTotal Hari Stockout per SKU (semua periode, semua toko):")
        lines.append(sku_total.to_string(index=False))

    # ── 6. Stockout per SKU per Tahun ────────────────────────────────────────
    if has_flag and has_year:
        sku_yr = (
            filtered_df.groupby(grp_sku + ["year"], observed=True)["stock_out_flag"]
            .sum().reset_index()
            .rename(columns={"stock_out_flag": "stockout_days"})
            .sort_values(["sku_id", "year"])
        )
        lines.append("\nStockout per SKU per Tahun (hari, semua toko digabung):")
        lines.append(sku_yr.to_string(index=False))

    # Top 40 SKU by total stockout — dihitung sekali, dipakai di bagian 7 & 8
    # agar context tidak terlalu besar dan groupby tidak diulang.
    top_sku_ids = (
        filtered_df.groupby("sku_id", observed=True)["stock_out_flag"]
        .sum().nlargest(40).index.tolist()
    ) if has_flag else []

    # ── 7. [KUNCI] Stockout per Store per SKU per Bulan ─────────────────────
    # Data ini menjawab: "Store X bulan Y → SKU apa yang stockout?"
    # dan "SKU X → di store mana saja stockout-nya per bulan?"
    if has_flag and has_store and has_year and has_month:
        grp_sm = ["store_id", "sku_id"] + sku_name_cols + ["year", "month"]
        store_month = (
            filtered_df[filtered_df["sku_id"].isin(top_sku_ids)]
            .groupby(grp_sm, observed=True)["stock_out_flag"]
            .sum().reset_index()
            .rename(columns={"stock_out_flag": "stockout_days"})
        )
        store_month = store_month[store_month["stockout_days"] > 0].copy()
        if not store_month.empty:
            # Nama bulan Indonesia agar AI bisa mencocokkan "Juni" langsung,
            # tanpa harus memetakan angka bulan sendiri (sumber kesalahan).
            store_month["bulan"] = store_month["month"].map(_MONTH_MAP)
            out_cols = ["store_id", "sku_id"] + sku_name_cols + ["year", "month", "bulan", "stockout_days"]
            store_month = store_month.sort_values(
                ["store_id", "year", "month", "sku_id"]
            )[out_cols]
            lines.append(
                "\nStockout per Toko per SKU per Bulan — hanya baris dengan stockout_days > 0\n"
                "(kolom 'bulan' = nama bulan, 'month' = angka bulan. "
                "Gunakan tabel ini untuk menjawab: store X bulan Y → SKU apa? "
                "atau SKU X → store mana bulan apa?):"
            )
            lines.append(store_month.to_string(index=False))

    # ── 8. Stockout per SKU per Toko per Tahun (ringkasan tahunan per store) ─
    if has_flag and has_store and has_year:
        grp_sty = ["sku_id", "store_id"] + sku_name_cols + ["year"]
        sku_store_yr = (
            filtered_df[filtered_df["sku_id"].isin(top_sku_ids)]
            .groupby(grp_sty, observed=True)["stock_out_flag"]
            .sum().reset_index()
            .rename(columns={"stock_out_flag": "stockout_days"})
            .query("stockout_days > 0")
            .sort_values(["sku_id", "store_id", "year"])
        )
        if not sku_store_yr.empty:
            lines.append("\nStockout per SKU per Toko per Tahun — hanya baris > 0:")
            lines.append(sku_store_yr.to_string(index=False))

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

        "ATURAN WAJIB — cara membaca dan menyajikan data:\n"
        "1. JANGAN tampilkan nilai numerik risk_score. Cukup tampilkan risk_level: "
        "Tinggi / Menengah / Rendah. (risk_score adalah skor relatif persentil — "
        "nilai kecil pun bisa 'Tinggi' jika termasuk tertinggi di dataset.)\n"
        "2. JANGAN menjumlahkan data bulan secara manual — gunakan tabel yang sudah dihitung.\n"
        "3. JANGAN tampilkan rumus penjumlahan seperti '12 + 11 + ... = 151'.\n"
        "4. Data mencakup 3 tahun historis (2021, 2022, 2023).\n\n"

        "PANDUAN MENJAWAB BERDASARKAN TIPE PERTANYAAN:\n"
        "A. Pertanyaan 'Store X, bulan Y → SKU apa yang stockout?':\n"
        "   → Filter tabel 'Stockout per Toko per SKU per Bulan' pada store_id=X, "
        "year=Y, month=Z, lalu tampilkan semua SKU yang muncul (stockout_days > 0).\n"
        "   → Jika tidak ada baris yang cocok, jawab 'tidak ada stockout tercatat'.\n"
        "   → JANGAN mengarang atau mengasumsikan — hanya laporkan yang ada di tabel.\n\n"
        "B. Pertanyaan 'SKU X → di store mana saja? detail per store?':\n"
        "   → Bagian 1 — Ringkasan: dari 'Total Hari Stockout per SKU' dan "
        "'Stockout per SKU per Tahun'.\n"
        "   → Bagian 2 — Tabel per toko: gabungkan 'Kondisi Terkini SKU-Toko' (stok, risk_level) "
        "dengan 'Stockout per SKU per Toko per Tahun' untuk kolom stockout per tahun. "
        "Tampilkan: Toko | Risk Level | Stok Saat Ini | Sales 7hr | Stockout 2021 | 2022 | 2023.\n"
        "   → Bagian 3 — Rekomendasi toko mana yang paling mendesak.\n\n"
        "C. Pertanyaan tentang KATEGORI:\n"
        "   → Gunakan tabel 'Risiko per Kategori' dan 'Total Hari Stockout per Kategori'.\n"
        "   → JANGAN gunakan angka total keseluruhan dashboard untuk satu kategori.\n\n"

        "Gunakan bahasa Indonesia yang jelas dan ringkas. Gunakan tabel markdown untuk "
        "data numerik. Jawab HANYA berdasarkan data yang diberikan di bawah ini.\n\n"
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
                    # Bangun konteks (termasuk inferensi model) HANYA saat cache miss.
                    # Saat cache hit, latest_df tidak diperlukan sama sekali.
                    if "_cached_context" not in st.session_state:
                        latest_df = _get_latest_with_risk(filtered_df, model)
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
