import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Exact feature order the model was trained on (42 features).
# Column order MUST match training. Do not change without retraining.
# ---------------------------------------------------------------------------

RAW_FEATURES = [
    "stock_on_hand",
    "units_sold",
    "lead_time_days",
    "discount_pct",
    "promo_flag",
    "is_weekend",
    "is_holiday",
]

CORE_ENGINEERED = [
    "stock_velocity",
    "lead_time_risk",
    "promo_discount_interaction",
    "log_sales",
    "danger_zone",
    "is_critical_stock",
]

TIMESERIES_FEATURES = [
    "rolling_7d_sales",
    "rolling_14d_sales",
    "lag_1_stock",
    "stock_change",
    "sales_trend",
]

ADVANCED_FEATURES = [
    "stock_coverage_days",
    "is_critical_coverage",
    "sales_volatility",
    "velocity_change",
    "lead_time_coverage_ratio",
    "promo_sales_ratio",
    "weekend_holiday",
]

# One-hot encoding columns (drop_first=True during training)
WEEKDAY_DUMMIES = [f"weekday_{i}" for i in range(1, 7)]   # weekday_1..6
MONTH_DUMMIES   = [f"month_{i}"   for i in range(2, 13)]  # month_2..12

FEATURE_ORDER = (
    RAW_FEATURES
    + CORE_ENGINEERED
    + TIMESERIES_FEATURES
    + ADVANCED_FEATURES
    + WEEKDAY_DUMMIES
    + MONTH_DUMMIES
)  # total = 7 + 6 + 5 + 7 + 6 + 11 = 42


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_feature_engineering(df: pd.DataFrame):
    """
    Replicate exact feature engineering pipeline used during training.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all raw columns from the CSV.
        Should include full history per SKU for rolling features to be accurate.

    Returns
    -------
    X : np.ndarray of shape (n_rows, 42)
    df_fe : pd.DataFrame with all engineered columns retained
            (needed for days_left, rolling_7d_sales, etc. in post-processing)
    """
    df_fe = df.copy()

    # Ensure date is datetime and sort for rolling correctness
    if "date" in df_fe.columns:
        df_fe["date"] = pd.to_datetime(df_fe["date"])

    id_col = "sku_id" if "sku_id" in df_fe.columns else "product_id"
    df_fe = df_fe.sort_values([id_col, "date"]).reset_index(drop=True)

    # ---- CORE FEATURES ----
    days_to_stockout = df_fe["stock_on_hand"] / (df_fe["units_sold"] + 0.1)

    df_fe["stock_velocity"] = df_fe["units_sold"] / (df_fe["stock_on_hand"] + 1)
    df_fe["lead_time_risk"] = df_fe["lead_time_days"] / (
        days_to_stockout.clip(upper=365) + 1
    )
    df_fe["promo_discount_interaction"] = (
        df_fe["promo_flag"] * df_fe["discount_pct"]
    )
    df_fe["log_sales"] = np.log1p(df_fe["units_sold"])

    is_low_stock = (days_to_stockout < 7).astype(int)
    df_fe["danger_zone"] = (
        (df_fe["stock_velocity"] > df_fe["stock_velocity"].quantile(0.75))
        & (is_low_stock == 1)
    ).astype(int)

    df_fe["is_critical_stock"] = (days_to_stockout < 3).astype(int)

    # ---- TIME-SERIES FEATURES ----
    df_fe["rolling_7d_sales"] = df_fe.groupby(id_col)["units_sold"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df_fe["rolling_14d_sales"] = df_fe.groupby(id_col)["units_sold"].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )
    df_fe["lag_1_stock"] = (
        df_fe.groupby(id_col)["stock_on_hand"]
        .shift(1)
        .fillna(df_fe["stock_on_hand"])
    )
    df_fe["stock_change"] = df_fe["stock_on_hand"] - df_fe["lag_1_stock"]
    df_fe["sales_trend"] = (
        df_fe.groupby(id_col)["units_sold"].diff().fillna(0)
    )

    # ---- ADVANCED FEATURES ----
    df_fe["stock_coverage_days"] = df_fe["stock_on_hand"] / (
        df_fe["rolling_7d_sales"] + 0.1
    )
    df_fe["is_critical_coverage"] = (
        df_fe["stock_coverage_days"] < 3
    ).astype(int)

    df_fe["sales_volatility"] = df_fe.groupby(id_col)["units_sold"].transform(
        lambda x: (
            x.rolling(14, min_periods=1).std()
            / (x.rolling(14, min_periods=1).mean() + 1)
        )
    )
    df_fe["velocity_change"] = (
        df_fe.groupby(id_col)["stock_velocity"].diff().fillna(0)
    )
    df_fe["lead_time_coverage_ratio"] = df_fe["lead_time_days"] / (
        df_fe["stock_coverage_days"] + 1
    )

    _lag_sales = (
        df_fe.groupby(id_col)["units_sold"]
        .shift(1)
        .fillna(df_fe["units_sold"])
    )
    df_fe["promo_sales_ratio"] = (
        df_fe["units_sold"] / (_lag_sales + 1)
    ) * df_fe["promo_flag"]

    df_fe["weekend_holiday"] = df_fe["is_weekend"] * df_fe["is_holiday"]

    # ---- ONE-HOT ENCODING ----
    weekday_dummies = pd.get_dummies(
        df_fe["weekday"], prefix="weekday", drop_first=True, dtype=int
    )
    month_dummies = pd.get_dummies(
        df_fe["month"], prefix="month", drop_first=True, dtype=int
    )

    # Reindex to ensure all expected dummy columns exist (handles sparse filters)
    weekday_dummies = weekday_dummies.reindex(
        columns=WEEKDAY_DUMMIES, fill_value=0
    )
    month_dummies = month_dummies.reindex(
        columns=MONTH_DUMMIES, fill_value=0
    )

    # ---- ASSEMBLE IN EXACT ORDER ----
    base_cols = RAW_FEATURES + CORE_ENGINEERED + TIMESERIES_FEATURES + ADVANCED_FEATURES
    X_df = pd.concat(
        [df_fe[base_cols].reset_index(drop=True),
         weekday_dummies.reset_index(drop=True),
         month_dummies.reset_index(drop=True)],
        axis=1,
    )

    # Sanity check
    assert X_df.shape[1] == 42, (
        f"Feature count mismatch: expected 42, got {X_df.shape[1]}"
    )

    # Fill any NaN that may arise from rolling/diff on short histories
    X_df = X_df.fillna(0.0)

    # Return named DataFrame so sklearn doesn't warn about missing feature names
    # (model was trained without names, so .values is ultimately passed)
    X = X_df.values.astype(float)

    return X, df_fe  # return df_fe so callers can access rolling_7d_sales etc.
