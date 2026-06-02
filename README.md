# FMCG Stockout Risk Prediction Dashboard

> **Final Project Machine Learning — Semester 6**
> Sistem prediksi dan visualisasi risiko kehabisan stok (stockout) untuk produk FMCG berbasis Streamlit, dilengkapi AI Assistant berbasis OpenAI.

---

## Gambaran Umum

Dashboard ini adalah **aplikasi web interaktif berbasis Streamlit** yang membantu manajer inventori mengidentifikasi:
- SKU mana yang **berisiko tinggi** mengalami stockout
- **Kapan** stok tersebut diprediksi akan habis
- **Tindakan apa** yang perlu diambil (restock, order, atau monitor)
- **Analisis mendalam** via AI Assistant yang memahami data inventori secara langsung

### Komponen Utama Sistem:
| Komponen | Deskripsi |
|---|---|
| Model ML | Calibrated LightGBM yang sudah dilatih sebelumnya |
| Dataset | 1.1 juta baris data historis selama 3 tahun (2021–2023) |
| Feature Engineering | 42 fitur konsisten dengan pipeline training |
| Visualisasi | Grafik interaktif berbasis Plotly |
| Simulasi | Proyeksi risiko stockout ke masa depan |
| AI Assistant | Chatbot berbasis GPT-4o-mini dengan konteks data aktual |

---

## Struktur Proyek

```
finalProject/
├── app.py                          → Entry point utama Streamlit
├── requirements.txt                → Daftar library yang dibutuhkan
├── README.md                       → Dokumentasi proyek (file ini)
├── convert_to_parquet.py           → Script konversi CSV → Parquet (jalankan sekali)
│
├── components/                     → Modul UI terpisah per fitur
│   ├── filters.py                  → Panel filter sidebar (+ SKU search autocomplete)
│   ├── overview.py                 → Ringkasan risiko + proyeksi masa depan
│   ├── prediction.py               → Prediksi mendalam untuk satu SKU
│   ├── visualization.py            → Chart historis + forecast 7 hari
│   ├── action_table.py             → Tabel aksi yang bisa diekspor
│   └── chatbot.py                  → AI Assistant (OpenAI streaming + konteks data)
│
├── utils/                          → Utilitas backend
│   ├── helpers.py                  → Load data, load model, business logic
│   └── feature_engineering.py     → Replikasi pipeline FE dari training
│
├── model/
│   └── calibrated_lgbm.pkl         → Model LightGBM yang dikalibrasi (~29 MB)
│
└── data/
    ├── fmcg_sales_3years_1M_rows.csv     → Dataset historis (211 MB, fallback)
    └── fmcg_sales_3years_1M_rows.parquet → Dataset terkompresi (14 MB, primary)
```

---

## Instalasi & Menjalankan Aplikasi

### 1. Clone / Siapkan Proyek

Pastikan semua file tersedia termasuk `model/calibrated_lgbm.pkl` dan file data di folder `data/`.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Opsional) Konversi Data ke Parquet

Jika file `.parquet` belum ada, jalankan konversi sekali untuk mempercepat loading:

```bash
python convert_to_parquet.py
```

> Jika dilewati, aplikasi akan otomatis mengkonversi saat pertama kali dijalankan (lebih lambat ~15–30 detik) dan menyimpan hasilnya untuk sesi berikutnya.

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

### 5. Buka di Browser

```
http://localhost:8501
```

---

## Cara Penggunaan

1. **Filter di Sidebar** — Pilih Store, Category, Date Range yang ingin dianalisis.
2. **Cari SKU** — Di mode Single SKU, ketik nama atau ID SKU untuk mendapatkan rekomendasi pencarian otomatis.
3. **Tab Dashboard** — Lihat ringkasan KPI risiko, proyeksi masa depan, dan tabel aksi.
4. **Tab AI Assistant** — Tanyakan analisis langsung dalam bahasa Indonesia; AI menjawab berdasarkan data aktual yang sedang ditampilkan.
5. **Single SKU Mode** — Pilih mode "Single SKU" di sidebar untuk analisis mendalam satu produk, termasuk chart historis stok dan forecast risiko 7 hari.

---

## Dokumentasi Flow

### 1. Flow Startup Aplikasi

```
streamlit run app.py
         │
         ▼
┌─────────────────────────────────────┐
│          load_data()                │
│  ┌──────────────────────────────┐   │
│  │ Parquet ada & valid?         │   │
│  │  Ya → pd.read_parquet() ✓   │   │  @st.cache_data
│  │  Tidak/korup →               │   │  (1x per sesi)
│  │    baca CSV chunks 100k/iter │   │
│  │    konversi dtype            │   │
│  │    simpan Parquet            │   │
│  └──────────────────────────────┘   │
│  Output: df (1.1 juta baris)        │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│          load_model()               │  @st.cache_resource
│  joblib.load(calibrated_lgbm.pkl)  │  (1x per sesi)
│  Output: CalibratedClassifierCV    │
└─────────────────────────────────────┘
         │
         ▼
   render_filters(df)
   [lihat Flow 2]
         │
         ▼
   st.tabs(["Dashboard", "AI Assistant"])
   ┌──────────────┬──────────────────┐
   │  tab_ai      │  tab_dashboard   │
   │  (render     │  (render kedua,  │
   │   pertama)   │   lebih lambat)  │
   └──────────────┴──────────────────┘
```

---

### 2. Flow Filter Sidebar

```
render_filters(df)
      │
      ├─► Store selectbox
      │       "All Stores" atau store_id tertentu
      │
      ├─► Category multiselect
      │       default: semua kategori
      │
      ├─► Analysis Mode radio
      │       ┌─────────────────┬──────────────────────┐
      │       │ Bulk (All SKUs) │ Single SKU            │
      │       │                 │                       │
      │       │ selected_sku    │ ┌─────────────────┐   │
      │       │ = None          │ │ text_input       │   │
      │       │                 │ │ "Cari SKU"       │   │
      │       │                 │ │                  │   │
      │       │                 │ │ ada query?       │   │
      │       │                 │ │  Ya → filter +   │   │
      │       │                 │ │       radio      │   │
      │       │                 │ │  Tidak → full    │   │
      │       │                 │ │          selectbox│  │
      │       │                 │ │                  │   │
      │       │                 │ │ no result?       │   │
      │       │                 │ │  → st.error()    │   │
      │       │                 │ │  → st.stop() ✋  │   │
      │       │                 └─────────────────┘   │
      │       └─────────────────┴──────────────────────┘
      │
      ├─► Date Range date_input
      │
      ▼
Apply Filters:
  df  →  [store filter]  →  [category filter]  →  [sku filter]  →  [date filter]
                                                                          │
                                                    filtered.empty? ──── ▼
                                                         Ya → st.error + st.stop ✋
                                                         Tidak → return (filtered_df, selected_sku)
```

---

### 3. Flow Dashboard Tab

```
with tab_dashboard:
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│                  render_overview(filtered_df, model)      │
│                                                           │
│  n_sku_est > 5000? → tampilkan warning                   │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │         compute_batch_prediction()                  │  │
│  │         @st.cache_data (hash: _df_fingerprint)     │  │
│  │                                                     │  │
│  │  Cache HIT  ─────────────────────────► return ✓    │  │
│  │  Cache MISS:                                        │  │
│  │    run_feature_engineering(filtered_df)             │  │
│  │      → 42 fitur per baris                          │  │
│  │    model.predict_proba(X)[:,1]                      │  │
│  │      → risk_score [0.0 – 1.0] per baris            │  │
│  │    groupby(sku_id, store_id).last()                 │  │
│  │      → latest_df (1 baris per SKU-Store)            │  │
│  └────────────────────────────────────────────────────┘  │
│                        │                                  │
│                        ▼                                  │
│  compute_risk_thresholds(risk_score)                     │
│    high_thresh   = percentile 95                         │
│    medium_thresh = percentile 75                         │
│  → simpan ke session_state                               │
│                        │                                  │
│                        ▼                                  │
│  latest_df["risk_level"] = High / Medium / Low           │
│                        │                                  │
│                        ▼                                  │
│  Render KPI Cards: Total | High | Medium | Low | AvgScore│
│  Render Future Risk Projection (date range picker)        │
│  Render Cumulative Stockout Chart (Plotly)               │
└──────────────────────────────────────────────────────────┘
         │
         ▼
render_action_table(latest_df)
  → sort by risk_score DESC
  → filter: High Risk Only / Action filter
  → Export CSV button
         │
         ▼
  selected_sku ada? (Single SKU mode)
  ┌── Tidak → selesai
  └── Ya ──►
         │
         ▼
render_prediction(filtered_df, model, selected_sku)
  → filter ke satu SKU
  → FE pada seluruh riwayat (agar rolling features akurat)
  → predict_proba → risk_score terbaru
  → 4 metric cards: Probability | Days Left | Risk Level | Action
         │
         ▼
  session_state["analyzed"] == True?
  └── Ya ──►
         │
         ▼
  col_left: render_stock_chart(sku_df)
    → Line chart: stock_on_hand | units_sold | rolling_7d_sales

  col_right: render_risk_timeline(sku_df, model)
    → Simulasi 7 hari ke depan:
        for day in range(7):
          stock -= rolling_7d_sales / 7
          run FE + predict → risk_score[day]
    → Bar chart berwarna per hari (High/Medium/Low)
```

---

### 4. Flow AI Assistant Tab

```
with tab_ai:
  render_chatbot(filtered_df, model)
         │
         ▼
  ┌─────────────────────────────────────┐
  │  API Key sudah ada di session_state? │
  │  Tidak → tampilkan form input key   │
  │    [sk-...] [Aktifkan]              │
  │    → simpan ke session_state        │
  │    → tidak st.rerun() (tab tetap)  │
  └─────────────────────────────────────┘
         │ Ya (key sudah ada)
         ▼
  Deteksi perubahan filter:
    current_fp = _df_fingerprint(filtered_df)
    ┌── fp sama → gunakan cache yang ada
    └── fp beda → hapus _cached_context
                  reset _context_cached = False
         │
         ▼
  prompt = st.chat_input(...)   ← sticky di bawah halaman
         │
  ┌── Tidak ada input → render riwayat + welcome message
  │
  └── Ada input:
         │
         ▼
  append user message ke session_state["chat_messages"]
         │
         ▼
  ┌──────────────────────────────────────────────────────┐
  │  st.container(height=540)  ← scrollable              │
  │                                                       │
  │  Render semua riwayat pesan:                          │
  │    user    → kolom kanan (st.columns [1,3])           │
  │    assistant → kolom penuh kiri + avatar 🤖           │
  │                                                       │
  │  Pesan terakhir = user? → generate response:          │
  │  ┌─────────────────────────────────────────────────┐ │
  │  │  st.toast("⏳ Sedang memproses...")              │ │
  │  │  st.toast("📊 Data besar, mohon tunggu...")      │ │
  │  │  (hanya jika belum pernah diproses)              │ │
  │  │                                                   │ │
  │  │  _get_latest_with_risk(filtered_df, model)       │ │
  │  │    → compute_batch_prediction() [cached]         │ │
  │  │    → compute_risk_thresholds() [dari data aktual]│ │
  │  │    → tambahkan risk_level                        │ │
  │  │                                                   │ │
  │  │  _build_context(latest_df, filtered_df)          │ │
  │  │  [cache di session_state["_cached_context"]]     │ │
  │  │    → ringkasan risiko per kategori               │ │
  │  │    → top 10 SKU risiko tertinggi (model)         │ │
  │  │    → top 20 SKU total hari stockout (historis)   │ │
  │  │    → top 5 SKU stockout per bulan per tahun      │ │
  │  │    → tingkat stockout % per bulan                │ │
  │  │                                                   │ │
  │  │  _stream_openai(api_key, history, context)       │ │
  │  │    → system_prompt + context + riwayat chat      │ │
  │  │    → GPT-4o-mini streaming                       │ │
  │  │    → st.write_stream() → teks muncul bertahap    │ │
  │  └─────────────────────────────────────────────────┘ │
  │                                                       │
  │  append assistant response ke session_state           │
  └──────────────────────────────────────────────────────┘
```

---

### 5. Flow Cache & Invalidasi

```
Setiap Streamlit rerun (widget berubah, chat dikirim, dll):
         │
         ▼
_df_fingerprint(filtered_df)
  = (len, date_min, date_max, sku_nunique, store_sample[8], cat_sample)
         │
         ├─► Digunakan oleh @st.cache_data pada compute_batch_prediction
         │     Cache HIT  → skip FE + ML (~0.1 detik)
         │     Cache MISS → jalankan FE + ML (~5-30 detik)
         │
         └─► Digunakan oleh render_chatbot untuk invalidasi session_state
               fingerprint sama → _cached_context dipertahankan
               fingerprint beda →
                 _cached_context dihapus
                 _context_cached direset ke False
                 (konteks AI akan dibangun ulang pada pesan berikutnya)

┌──────────────────────────────────────────────┐
│  Yang menyebabkan cache MISS / invalidasi:   │
│  ✓ Ganti Store                               │
│  ✓ Tambah/kurangi Category                  │
│  ✓ Ubah Date Range                           │
│  ✓ Pilih SKU berbeda (Single SKU mode)       │
│                                              │
│  Yang TIDAK menyebabkan cache miss:          │
│  ✗ Mengetik di chat input                   │
│  ✗ Klik Reset chat                          │
│  ✗ Scroll halaman                           │
│  ✗ Klik tab Dashboard ↔ AI Assistant        │
└──────────────────────────────────────────────┘
```

---

### 6. Flow Data Loading (Parquet vs CSV)

```
load_data() dipanggil saat startup
         │
         ▼
  data/fmcg_sales_3years_1M_rows.parquet ada?
         │
    ┌────┴────┐
   Ya        Tidak
    │          │
    ▼          ▼
  Baca      Baca CSV (chunked, 100k baris/iter)
  Parquet     → cast dtype (int8/16/32, float32, category)
  (~1-2 dtk)  → pd.concat semua chunk
    │          → simpan Parquet untuk sesi berikutnya
    │               (try/except: gagal ≠ crash)
    ▼          ↓
  df siap   (~15-30 detik, hanya sekali)
    │
    ▼
  Parquet korup saat baca?
    → hapus file korup
    → fallback ke CSV otomatis
```

---

### 7. Flow Lengkap Per Interaksi

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SETIAP STREAMLIT RERUN                            │
│                                                                      │
│  load_data()     ──── @st.cache_data ──── return df (instan)        │
│  load_model()    ── @st.cache_resource ── return model (instan)     │
│                                                                      │
│  render_filters(df)                                                  │
│    → baca widget values dari session_state                           │
│    → apply filters → filtered_df                                     │
│                                                                      │
│  ┌─── tab_ai (render lebih dulu) ───────────────────────────────┐   │
│  │  render_chatbot(filtered_df, model)                           │   │
│  │    → cek API key (instan)                                     │   │
│  │    → cek fingerprint (instan)                                 │   │
│  │    → render riwayat chat (instan)                             │   │
│  │    → jika ada prompt baru:                                    │   │
│  │        compute context (cached/~5-30dtk saat pertama)        │   │
│  │        stream response dari OpenAI                            │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─── tab_dashboard (render setelah tab_ai) ──────────────────┐     │
│  │  render_overview(filtered_df, model)                        │     │
│  │    → compute_batch_prediction (cached/~5-30dtk pertama)    │     │
│  │    → hitung threshold → render KPI + projection chart      │     │
│  │  render_action_table(latest_df)                             │     │
│  │    → render tabel interaktif                                │     │
│  │  [jika Single SKU] render_prediction + visualization       │     │
│  └─────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Penjelasan Komponen

### Filters (`components/filters.py`)

Sidebar kiri dengan filter berikut:

| Filter | Deskripsi |
|---|---|
| Store | Dropdown satu toko atau "All Stores" |
| Category | Multi-select kategori produk |
| Mode | Bulk (semua SKU) atau Single SKU |
| Cari SKU | Text input dengan rekomendasi pencarian real-time (cocokkan ID dan nama SKU) |
| Date Range | Rentang tanggal data yang dianalisis |

**SKU Search Autocomplete:**
- Ketik sebagian nama atau ID SKU → tampil daftar rekomendasi (maks 10 hasil)
- Pencarian tidak case-sensitive, cocok pada `sku_id` maupun `sku_name`
- Jika pencarian tidak menemukan hasil, sistem menampilkan pesan error dan menghentikan eksekusi agar ML pipeline tidak berjalan pada data yang salah

---

### Overview (`components/overview.py`)

**Bagian A — Risk Overview (Current State):**
- Memanggil `compute_batch_prediction` (hasil di-cache) → FE + prediksi model
- Mengambil baris terbaru per SKU-Store
- Menampilkan **5 KPI Card**: Total SKUs | High Risk | Medium Risk | Low Risk | Avg Risk Score

**Bagian B — Future Risk Projection:**
- Menghitung `days_left = stock_on_hand / rolling_7d_sales`
- Menghitung `predicted_stockout_date` dan `restock_needed_by`
- User memilih rentang tanggal proyeksi (bisa hingga tahun depan)

KPI Proyeksi:
| Metrik | Deskripsi |
|---|---|
| Stockout Sebelum Window | SKU yang sudah habis sebelum tanggal awal |
| Stockout Dalam Periode | SKU yang habis antara dua tanggal |
| Harus Order Dalam Periode | SKU yang harus dipesan dalam rentang ini |
| Aman Sampai Akhir Period | SKU yang masih aman hingga tanggal akhir |

---

### Action Table (`components/action_table.py`)

Tabel yang diurutkan berdasarkan `risk_score` (tertinggi di atas).

**Kolom tersedia:**
- `sku_id`, `sku_name`, `store_id`, `category`
- Risk Level (dengan emoji), Risk Score (progress bar)
- Days Left, Predicted Stockout Date, Restock Needed By
- Stock On Hand, Rolling Avg Sales, Lead Time
- Suggested Order Qty, Action

**Filter & Export:**
- Checkbox "High Risk Only"
- Selectbox "Action Filter"
- Tombol "Export to CSV"

---

### AI Assistant (`components/chatbot.py`)

Chatbot interaktif berbasis **GPT-4o-mini** dengan streaming real-time.

**Fitur:**
- Input API key OpenAI sekali per sesi (disimpan di `session_state`, tidak ke disk)
- Konteks data dikirim otomatis sesuai filter aktif:
  - Ringkasan risiko per kategori (dari prediksi model LightGBM)
  - Top 20 SKU berdasarkan total hari stockout historis
  - Top 5 SKU stockout per bulan per tahun (2021–2023)
  - Tingkat stockout per bulan dalam persentase
- Riwayat percakapan dipertahankan dalam sesi
- Konteks di-cache — tidak dihitung ulang setiap pesan (hanya saat filter berubah)
- Streaming response seperti Claude/ChatGPT

**Contoh pertanyaan yang bisa dijawab:**
- "SKU mana yang harus segera direstock minggu ini?"
- "Berapa banyak SKU berisiko tinggi di kategori Beverages?"
- "Setiap bulan Januari, SKU apa yang paling sering stockout?"
- "Toko mana yang paling banyak SKU berbahaya?"

---

### Prediction (`components/prediction.py`)

Aktif hanya di **Single SKU Mode**.

Alur saat tombol "Analyze SKU" diklik:
1. Filter data untuk SKU yang dipilih
2. Jalankan FE pada seluruh riwayat SKU (agar rolling features akurat)
3. Jalankan model → risk_score per baris
4. Tampilkan kondisi terbaru dalam 4 metric card:
   - **Stockout Probability** | **Days Until Stockout** | **Risk Level** | **Action**

---

### Visualization (`components/visualization.py`)

**Stock Chart** — Line chart interaktif:
- Stock On Hand (hijau, fill ke bawah)
- Units Sold (merah)
- 7-Day Rolling Avg Sales (kuning, putus-putus)

**Risk Timeline (7-Day Forecast)** — Simulasi 7 hari ke depan:
- Setiap hari: kurangi stok dengan rata-rata penjualan harian
- Jalankan FE + model pada data gabungan
- Tampilkan 7 skor terakhir sebagai prediksi per hari
- Kode warna: High | Medium | Low
- Garis referensi horizontal di 0.5 (Medium) dan 0.8 (High)

---

## Feature Engineering

Pipeline FE adalah **replikasi persis** dari pipeline training. Total **42 fitur** dengan urutan identik.

### Fitur Mentah (7)
`stock_on_hand`, `units_sold`, `lead_time_days`, `discount_pct`, `promo_flag`, `is_weekend`, `is_holiday`

### Fitur Inti Engineered (6)
| Fitur | Formula |
|---|---|
| `stock_velocity` | `units_sold / (stock_on_hand + 1)` |
| `lead_time_risk` | `lead_time_days / (days_to_stockout + 1)` |
| `promo_discount_interaction` | `promo_flag × discount_pct` |
| `log_sales` | `log(1 + units_sold)` |
| `danger_zone` | `1` jika velocity tinggi DAN stok < 7 hari |
| `is_critical_stock` | `1` jika stok < 3 hari |

### Fitur Time-Series (5)
`rolling_7d_sales`, `rolling_14d_sales`, `lag_1_stock`, `stock_change`, `sales_trend`

### Fitur Lanjutan (7)
`stock_coverage_days`, `is_critical_coverage`, `sales_volatility`, `velocity_change`, `lead_time_coverage_ratio`, `promo_sales_ratio`, `weekend_holiday`

### One-Hot Encoding (17)
- `weekday_1` ... `weekday_6` (6 kolom, drop_first=True)
- `month_2` ... `month_12` (11 kolom, drop_first=True)

> **Total: 7 + 6 + 5 + 7 + 6 + 11 = 42 fitur ✓**

---

## Model & Threshold

### Spesifikasi Model

| Properti | Detail |
|---|---|
| Jenis | `CalibratedClassifierCV` (sklearn) |
| Base Model | LightGBM Classifier |
| File | `model/calibrated_lgbm.pkl` (~29 MB) |
| Output | `predict_proba(X)[:,1]` — probabilitas stockout |
| Jumlah Fitur | 42 (Column_0 hingga Column_41) |

### Dynamic Thresholds (Percentile-Based)

Karena model dikalibrasi dengan base rate rendah (~3%), threshold absolut seperti 0.5 atau 0.8 **tidak efektif** — hampir semua SKU akan terklasifikasi sebagai Low.

**Solusi:** Threshold dihitung dari distribusi aktual batch yang sedang dianalisis:

| Level | Threshold |
|---|---|
| High | Skor ≥ persentil ke-95 (top 5% SKU) |
| Medium | Skor ≥ persentil ke-75 (top 25% SKU) |
| Low | Skor < persentil ke-75 |

> Threshold dihitung secara konsisten dari data aktual untuk setiap komponen (Dashboard dan AI Assistant), sehingga klasifikasi risiko selalu seragam.

---

## Optimasi Performa

Dataset 1.1 juta baris (211 MB CSV) memerlukan penanganan khusus:

### A. Parquet sebagai Format Utama
- Dataset disimpan dalam format **Apache Parquet** (14 MB vs 211 MB CSV — hemat 93%)
- Loading: **~1–2 detik** vs ~15–30 detik untuk CSV
- Dtype tersimpan otomatis — tidak perlu konversi manual
- Fallback ke CSV otomatis jika file Parquet belum ada atau korup

### B. Dtype Optimization (pada CSV fallback)
- `int64` → `int8/int16/int32`
- `float64` → `float32`
- `string` → `category` (dictionary encoding)
- Penghematan: ~60–70% memory

### C. Streamlit Caching
- `@st.cache_data` untuk `load_data()` — data hanya dibaca **1x per sesi**
- `@st.cache_resource` untuk `load_model()` — model hanya di-load **1x per sesi**
- `@st.cache_data` untuk `compute_batch_prediction()` — ML pipeline tidak diulang saat interaksi chat atau UI lain selama filter tidak berubah

### D. Lazy Computation di AI Assistant
- Konteks data untuk AI dihitung **hanya saat user mengirim pesan pertama**
- Hasil konteks di-cache di `session_state` dan hanya diperbarui saat filter berubah
- AI Assistant dapat diakses segera tanpa menunggu Dashboard selesai memuat

### E. Batch Prediction
- Model **tidak** dijalankan pada seluruh 1.1 juta baris
- FE hanya dijalankan pada `filtered_df` (subset berdasarkan filter user)
- Setelah prediksi, hanya ambil baris terbaru per SKU-Store

---

## Limitasi & Catatan

1. **Data Historis**: Dataset berakhir sekitar akhir 2023. Proyeksi "masa depan" menggunakan kondisi stok terakhir sebagai baseline.

2. **Akurasi Model**: Skor **relatif (ranking)** lebih bermakna daripada nilai absolut. Model membedakan SKU berisiko tinggi vs rendah dalam satu batch.

3. **Simulasi Future**:
   - Menggunakan **linear depletion** (stok berkurang rata-rata penjualan per hari)
   - Tidak memperhitungkan restock/replenishment yang akan datang
   - Tidak memperhitungkan fluktuasi demand musiman

4. **Threshold Dinamis**: Threshold berubah setiap kali filter berubah — ini **disengaja** agar selalu ada SKU di kategori High dan Medium pada setiap batch analisis.

5. **AI Assistant**: Membutuhkan API key OpenAI milik pengguna. Key hanya tersimpan selama sesi browser aktif dan tidak disimpan ke server manapun.

---

## Dependencies

```txt
streamlit    >= 1.32.0   # Framework web dashboard
pandas       >= 2.0.0    # Manipulasi data
numpy        >= 1.24.0   # Komputasi numerik
lightgbm     >= 4.0.0    # Model machine learning
scikit-learn >= 1.3.0    # Calibration wrapper, training utilities
joblib       >= 1.3.0    # Load/save model
plotly       >= 5.18.0   # Visualisasi interaktif
pyarrow      >= 12.0.0   # Format Parquet (fast data loading)
openai       >= 1.0.0    # AI Assistant (opsional, butuh API key)
```

Install sekaligus:
```bash
pip install -r requirements.txt
```

---

## Informasi Proyek

| Keterangan | Detail |
|---|---|
| Mata Kuliah | Machine Learning |
| Semester | 6 |
| Tahun Akademik | 2025/2026 |
| Tanggal Pembuatan | 2026-04-01 |
| Dataset | FMCG Sales 3 Years (1.1 Juta Baris) |
| Model | Calibrated LightGBM |

---

> *Dashboard ini dirancang untuk mendukung pengambilan keputusan berbasis data dalam manajemen inventori produk FMCG.*
