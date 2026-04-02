import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------------------------
# Data & Model Loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load CSV in chunks to avoid peak-RAM spike during pandas tokenization.
    Each 100k-row chunk is cast to compact dtypes before accumulation.
    Final memory footprint: ~200 MB instead of ~1 GB.
    """
    dtype_map = {
        "year":        "int16",
        "month":       "int8",
        "day":         "int8",
        "weekofyear":  "int8",
        "weekday":     "int8",
        "is_weekend":  "int8",
        "is_holiday":  "int8",
        "promo_flag":  "int8",
        "stock_out_flag": "int8",
        "temperature": "float32",
        "rain_mm":     "float32",
        "latitude":    "float32",
        "longitude":   "float32",
        "list_price":  "float32",
        "discount_pct":"float32",
        "gross_sales": "float32",
        "net_sales":   "float32",
        "purchase_cost":"float32",
        "margin_pct":  "float32",
        "units_sold":  "int32",
        "stock_on_hand":"int32",
        "lead_time_days":"int16",
    }
    # String columns are read as object per-chunk, then converted after concat
    str_cols = [
        "store_id", "country", "city", "channel",
        "sku_id", "sku_name", "category", "subcategory",
        "brand", "supplier_id",
    ]

    chunks = []
    reader = pd.read_csv(
        "data/fmcg_sales_3years_1M_rows.csv",
        parse_dates=["date"],
        dtype=dtype_map,
        chunksize=100_000,   # process 100k rows at a time
        low_memory=False,
    )
    for chunk in reader:
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # Convert string columns to category after concat (memory efficient)
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


@st.cache_resource
def load_model():
    """Load calibrated LightGBM model once and cache as resource."""
    try:
        return joblib.load("model/calibrated_lgbm.pkl")
    except FileNotFoundError:
        st.error("Model file not found: model/calibrated_lgbm.pkl")
        st.stop()
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.stop()


# ---------------------------------------------------------------------------
# Business Logic
# ---------------------------------------------------------------------------

def compute_risk_thresholds(scores: "pd.Series") -> tuple:
    """
    Compute dynamic risk thresholds from the score distribution.
    Uses percentile-based approach so the dashboard always shows
    meaningful segmentation regardless of calibrated probability range.

    Returns (medium_thresh, high_thresh):
      - High  = top 5% of scores
      - Medium= 5th-25th percentile (top 5-25%)
      - Low   = bottom 75%
    """
    high_thresh   = float(scores.quantile(0.95))
    medium_thresh = float(scores.quantile(0.75))
    # Clamp: if all scores are identical, avoid both thresholds being equal
    if high_thresh <= medium_thresh:
        high_thresh = medium_thresh * 1.5 + 1e-9
    return medium_thresh, high_thresh


def get_risk_level(score: float, medium_thresh: float = 0.5,
                  high_thresh: float = 0.8) -> str:
    """
    Map probability score to risk label using dynamic or static thresholds.
    Call compute_risk_thresholds() first for a data-driven split.
    """
    if score >= high_thresh:
        return "High"
    elif score >= medium_thresh:
        return "Medium"
    else:
        return "Low"


def get_risk_emoji(risk_level: str) -> str:
    mapping = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    return mapping.get(risk_level, "⚪")


def get_days_left(stock_on_hand: float, rolling_7d_sales: float) -> float:
    """Estimate days of stock remaining based on 7-day rolling average sales."""
    import math
    if math.isnan(stock_on_hand) or math.isnan(rolling_7d_sales):
        return 365.0  # unknown stock → treat as safe
    days = stock_on_hand / (rolling_7d_sales + 0.1)
    return min(days, 365.0)  # cap to avoid inf display


def get_action(risk_level: str, days_left: float, lead_time: float) -> str:
    """Determine recommended action based on risk and urgency."""
    if risk_level == "High" and days_left < lead_time:
        return "Restock NOW"
    elif risk_level == "High":
        return "Order Soon"
    elif risk_level == "Medium":
        return "Monitor"
    else:
        return "Safe"


def suggest_order(rolling_7d_sales: float, lead_time: float,
                  safety_days: int = 3) -> int:
    """
    Suggest reorder quantity.
    Formula: cover lead time + safety buffer based on avg daily demand.
    """
    import math
    if math.isnan(rolling_7d_sales) or math.isnan(lead_time):
        return 0
    order_qty = rolling_7d_sales * (lead_time + safety_days)
    return max(0, int(order_qty))


# ---------------------------------------------------------------------------
# Dataset Helpers
# ---------------------------------------------------------------------------

def get_latest_per_sku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per (sku_id, store_id) — the most recent record.
    Used for batch overview to avoid running model on all 1M rows.
    """
    latest = (
        df.sort_values("date")
        .groupby(["sku_id", "store_id"], as_index=False)
        .last()
    )
    return latest.reset_index(drop=True)
