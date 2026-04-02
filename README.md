# FMCG Stockout Risk Prediction Dashboard

> **Final Project Machine Learning — Semester 6**
> Sistem prediksi dan visualisasi risiko kehabisan stok (stockout) untuk produk FMCG berbasis Streamlit.

---

## Gambaran Umum

Dashboard ini adalah **aplikasi web interaktif berbasis Streamlit** yang membantu manajer inventori mengidentifikasi:
- SKU mana yang **berisiko tinggi** mengalami stockout
- **Kapan** stok tersebut diprediksi akan habis
- **Tindakan apa** yang perlu diambil (restock, order, atau monitor)

### Komponen Utama Sistem:
| Komponen | Deskripsi |
|---|---|
| Model ML | Calibrated LightGBM yang sudah dilatih sebelumnya |
| Dataset | 1.1 juta baris data historis selama 3 tahun (2021–2023) |
| Feature Engineering | 42 fitur konsisten dengan pipeline training |
| Visualisasi | Grafik interaktif berbasis Plotly |
| Simulasi | Proyeksi risiko stockout ke masa depan |

---

## Struktur Proyek

```
finalProject/
├── app.py                          → Entry point utama Streamlit
├── requirements.txt                → Daftar library yang dibutuhkan
├── README.md                       → Dokumentasi proyek (file ini)
│
├── components/                     → Modul UI terpisah per fitur
│   ├── filters.py                  → Panel filter sidebar
│   ├── overview.py                 → Ringkasan risiko + proyeksi masa depan
│   ├── prediction.py               → Prediksi mendalam untuk satu SKU
│   ├── visualization.py            → Chart historis + forecast 7 hari
│   └── action_table.py             → Tabel aksi yang bisa diekspor
│
├── utils/                          → Utilitas backend
│   ├── helpers.py                  → Load data, load model, business logic
│   └── feature_engineering.py     → Replikasi pipeline FE dari training
│
├── model/
│   └── calibrated_lgbm.pkl         → Model LightGBM yang dikalibrasi (~29 MB)
│
└── data/
    └── fmcg_sales_3years_1M_rows.csv  → Dataset historis (211 MB)
```

---

## Instalasi & Menjalankan Aplikasi

### 1. Clone / Siapkan Proyek

Pastikan semua file sudah tersedia termasuk `model/calibrated_lgbm.pkl` dan `data/fmcg_sales_3years_1M_rows.csv`.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

### 4. Buka di Browser

```
http://localhost:8501
```

---

## 🚀 Cara Penggunaan

1. **Filter di Sidebar** — Pilih Store, Category, Date Range yang ingin dianalisis.
2. **Risk Overview** — Lihat ringkasan KPI risiko (Total SKU, High/Medium/Low Risk, Avg Risk Score).
3. **Future Risk Projection** — Tentukan rentang tanggal proyeksi untuk simulasi ke depan.
4. **Action Table** — Lihat rekomendasi tindakan per SKU, termasuk estimasi tanggal stockout dan jumlah order yang disarankan.
5. **Single SKU Mode** — Pilih mode "Single SKU" di sidebar untuk analisis mendalam satu produk.
6. **Analyze SKU** — Klik tombol "Analyze SKU" untuk melihat chart historis stok + forecast risiko 7 hari ke depan.

---

## Alur Data (End-to-End)

```
[User memilih filter di sidebar]
        ↓
[filtered_df: subset dari 1.1 juta baris]
        ↓
[Feature Engineering: 42 fitur dibuat dari data mentah]
        ↓
[Model Prediction: predict_proba → skor probabilitas stockout 0.0–1.0]
        ↓
[Ambil baris terbaru per (sku_id, store_id)]
        ↓
[Hitung Dynamic Thresholds dari distribusi aktual batch]
  → High   = top 5%  (persentil ke-95)
  → Medium = top 25% (persentil ke-75)
  → Low    = bawah 75%
        ↓
[Post-processing]
  → risk_level, days_left, suggested_order, action
        ↓
[Rendering UI: KPI cards, charts, action table, projections]
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
| SKU | Dropdown SKU (hanya di mode Single SKU) |
| Date Range | Rentang tanggal data yang dianalisis |

> Jika hasil filter kosong, sistem akan menampilkan error dan menghentikan eksekusi.

---

### Overview (`components/overview.py`)

**Bagian A — Risk Overview (Current State):**
- Menjalankan Feature Engineering + Model pada `filtered_df`
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
- 🟢 Stock On Hand (hijau, fill ke bawah)
- 🔴 Units Sold (merah)
- 🟡 7-Day Rolling Avg Sales (kuning, putus-putus)

**Risk Timeline (7-Day Forecast)** — Simulasi 7 hari ke depan:
- Setiap hari: kurangi stok dengan rata-rata penjualan harian
- Jalankan FE + model pada data gabungan
- Tampilkan 7 skor terakhir sebagai prediksi per hari
- Kode warna: 🔴 High | 🟡 Medium | 🟢 Low
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
| 🔴 High | Skor ≥ persentil ke-95 (top 5% SKU) |
| 🟡 Medium | Skor ≥ persentil ke-75 (top 25% SKU) |
| 🟢 Low | Skor < persentil ke-75 |

> Threshold disimpan di `session_state` dan digunakan konsisten di semua komponen.

---

## Optimasi Performa

Dataset 1.1 juta baris (211 MB) memerlukan penanganan khusus:

### A. Chunked CSV Loading
- Membaca CSV dalam potongan **100.000 baris per chunk**
- Setiap chunk langsung di-cast ke dtype optimal
- RAM usage: ~200 MB vs ~1 GB jika di-load sekaligus

### B. Dtype Optimization
- `int64` → `int8/int16/int32`
- `float64` → `float32`
- `string` → `category` (dictionary encoding)
- **Penghematan: ~60–70% memory**

### C. Streamlit Caching
- `@st.cache_data` untuk `load_data()` — CSV hanya dibaca **1x**
- `@st.cache_resource` untuk `load_model()` — model hanya di-load **1x**

### D. Batch Prediction
- Model **tidak** dijalankan pada seluruh 1.1 juta baris
- FE hanya dijalankan pada `filtered_df` (subset berdasarkan filter)
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

---

## Dependencies

```txt
streamlit   >= 1.32.0   # Framework web dashboard
pandas      >= 2.0.0    # Manipulasi data
numpy       >= 1.24.0   # Komputasi numerik
lightgbm    >= 4.0.0    # Model machine learning
scikit-learn >= 1.3.0   # Calibration wrapper, training utilities
joblib      >= 1.3.0    # Load/save model
plotly      >= 5.18.0   # Visualisasi interaktif
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
