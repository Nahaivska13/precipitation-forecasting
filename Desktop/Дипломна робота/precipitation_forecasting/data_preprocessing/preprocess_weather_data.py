"""
Weather Data Preprocessing Pipeline  v2
========================================
Prepares hourly weather data for precipitation intensity prediction models
(including Transformer). Handles 7 Ukrainian cities.

Improvements over v1:
  - Train/val/test split BEFORE normalization (no data leakage)
  - Scaler saved to disk (pickle) for inference reuse
  - Target (precip_mm) log-transformed + normalized with separate scaler
  - Sentinel-value detection (physically impossible values)
  - Temporal continuity check (missing hourly timestamps → reindex + interpolate)
  - Full missing-value report saved to JSON
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import ast
import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

CITIES = {
    "Kyiv":      {"lat": 50.45, "lon": 30.52},
    "Lviv":      {"lat": 49.84, "lon": 24.03},
    "Odesa":     {"lat": 46.48, "lon": 30.73},
    "Dnipro":    {"lat": 48.45, "lon": 35.05},
    "Kharkiv":   {"lat": 50.00, "lon": 36.23},
    "Uzhhorod":  {"lat": 48.62, "lon": 22.29},
    "Chernihiv": {"lat": 51.50, "lon": 31.30},
}
DIRECT_FEATURES = [
    "temp_c",       # air temperature
    "humidity",     # relative humidity
    "pressure_mb",  # atmospheric pressure
    "wind_kph",     # wind speed
    "wind_degree",  # wind direction
    "cloud",        # cloud cover %
    "dewpoint_c",   # dew point
    "vis_km",       # visibility
    "gust_kph",     # wind gusts
    "snow_cm",      # snow
    "precip_mm",    # TARGET
]

SENTINEL_BOUNDS = {
    "temp_c":       (-90,   60),
    "humidity":     (  0,  100),
    "pressure_mb":  (870, 1085),
    "wind_kph":     (  0,  400),
    "wind_degree":  (  0,  360),
    "cloud":        (  0,  100),
    "dewpoint_c":   (-90,   35),
    "vis_km":       (  0,  100),
    "gust_kph":     (  0,  500),
    "snow_cm":      (  0,  300),
    "precip_mm":    (  0,  500),
}

# ─── Train / val / test split ratios ─────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)


def extract_condition_code(val):
    """Parse condition dict string → numeric code."""
    try:
        if isinstance(val, str):
            d = ast.literal_eval(val)
            return int(d.get("code", 0))
    except Exception:
        pass
    return 0


def add_latent_features(df):
    """Cyclical time encodings + derived meteorological features."""
    df["datetime"] = pd.to_datetime(df["time"])
    hour  = df["datetime"].dt.hour
    month = df["datetime"].dt.month

    df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df["condition_code"]      = df["condition"].apply(extract_condition_code)
    df["temp_dewpoint_spread"] = df["temp_c"] - df["dewpoint_c"]

    wind_rad  = np.deg2rad(df["wind_degree"])
    df["wind_u"] = -df["wind_kph"] * np.sin(wind_rad)
    df["wind_v"] = -df["wind_kph"] * np.cos(wind_rad)

    return df


def add_lag_features(df):
    """
    Додає lag features для precip_mm.
    Модель бачить попередні значення опадів — найсильніший сигнал для прогнозу.
    Важливо: lag features додаються ДО split щоб NaN на початку були заповнені.
    """
    LAG_HOURS = [1, 2, 3, 6, 12]
    for lag in LAG_HOURS:
        col = f"precip_lag_{lag}h"
        df[col] = df["precip_mm"].shift(lag)

   
    lag_cols = [f"precip_lag_{lag}h" for lag in LAG_HOURS]
    df[lag_cols] = df[lag_cols].fillna(0.0)

    return df


def check_sentinel_values(df, city_name):
    """
    Replace physically impossible values with NaN.
    Returns (df, report_dict).
    """
    report = {}
    for col, (lo, hi) in SENTINEL_BOUNDS.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n_bad = mask.sum()
        if n_bad > 0:
            pct = n_bad / len(df) * 100
            print(f"    ⚠ [{city_name}] Sentinel in '{col}': "
                  f"{n_bad} values outside [{lo}, {hi}] ({pct:.2f}%) → set NaN")
            df.loc[mask, col] = np.nan
            report[col] = {"sentinel_count": int(n_bad), "pct": round(pct, 2),
                           "bounds": [lo, hi]}
    if not report:
        print(f"    ✓ [{city_name}] No sentinel values found.")
    return df, report

def check_temporal_continuity(df, city_name):
    """
    Ensure exactly one row per hour. Missing timestamps → reindex + interpolate.
    Returns (df, report_dict).
    """
    df = df.set_index("datetime").sort_index()

    expected = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing_ts = expected.difference(df.index)
    n_missing = len(missing_ts)

    report = {"missing_hours": n_missing}

    if n_missing == 0:
        print(f"    ✓ [{city_name}] Temporal continuity OK — no gaps.")
    else:
        pct = n_missing / len(expected) * 100
        print(f"    ⚠ [{city_name}] {n_missing} missing hourly timestamps "
              f"({pct:.2f}%) → reindex + interpolate")
        # Reindex to full hourly grid; new rows get NaN → interpolated below
        df = df.reindex(expected)
        report["pct"] = round(pct, 2)

    # Interpolate all numeric columns along the time axis
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")

    df = df.reset_index().rename(columns={"index": "datetime"})
    return df, report

def check_and_fill_missing(df, city_name):
    """
    Report NaN counts per column and fill:
      numeric  → linear interpolation → ffill → bfill
      non-numeric → ffill → bfill
    """
    total   = len(df)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    report  = {}

    if missing.empty:
        print(f"    ✓ [{city_name}] No NaN values.")
    else:
        print(f"    [{city_name}] NaN values:")
        for col, cnt in missing.items():
            pct = cnt / total * 100
            print(f"      • {col}: {cnt} ({pct:.2f}%)")
            report[col] = {"nan_count": int(cnt), "pct": round(pct, 2)}

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    remaining = df.isnull().sum().sum()
    if remaining == 0:
        print(f"    ✓ [{city_name}] All NaNs filled.")
    else:
        print(f"    ⚠ [{city_name}] {remaining} NaNs still remain!")
    return df, report

def temporal_split(df):
    """
    Chronological split — keeps time order intact.
    Returns (train_df, val_df, test_df).
    """
    n     = len(df)
    i_val  = int(n * TRAIN_RATIO)
    i_test = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:i_val].copy()
    val   = df.iloc[i_val:i_test].copy()
    test  = df.iloc[i_test:].copy()

    print(f"    Split → train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
    return train, val, test

def normalize_splits(train, val, test, city_name, output_dir):
    """
    Features : StandardScaler  fit on train → transform val & test
    Target   : log1p → StandardScaler  fit on train → transform val & test
    Scalers saved to disk.
    Returns (train, val, test, feature_scaler, target_scaler, norm_cols)
    """
    # Columns excluded from feature normalization
    exclude = {
        "precip_mm",  # target — handled separately
        "is_day",     # binary flag (already 0/1)
        "city",       # string
    }

    # ── Pre-normalization: clip snow_cm outliers ─────────────────────────────
    if "snow_cm" in train.columns:
        cap = train["snow_cm"].quantile(0.99)
        for split in (train, val, test):
            split["snow_cm"] = split["snow_cm"].clip(upper=cap)

    # ── Feature scaler (includes condition_code, lat, lon) ───────────────────
    norm_cols = [
        c for c in train.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    feat_scaler = StandardScaler()
    train[norm_cols] = feat_scaler.fit_transform(train[norm_cols])
    val[norm_cols]   = feat_scaler.transform(val[norm_cols])
    test[norm_cols]  = feat_scaler.transform(test[norm_cols])

    # ── Target: log1p transform then scale ──────────────────────────────────
    for split in (train, val, test):
        split["precip_mm"] = np.log1p(split["precip_mm"])

    tgt_scaler = StandardScaler()
    train[["precip_mm"]] = tgt_scaler.fit_transform(train[["precip_mm"]])
    val[["precip_mm"]]   = tgt_scaler.transform(val[["precip_mm"]])
    test[["precip_mm"]]  = tgt_scaler.transform(test[["precip_mm"]])

    print(f"    ✓ Normalized {len(norm_cols)} feature cols + target (log1p + scale).")

    # ── Save scalers ─────────────────────────────────────────────────────────
    scalers_dir = os.path.join(output_dir, "scalers")
    os.makedirs(scalers_dir, exist_ok=True)

    feat_path = os.path.join(scalers_dir, f"{city_name}_feature_scaler.pkl")
    tgt_path  = os.path.join(scalers_dir, f"{city_name}_target_scaler.pkl")

    with open(feat_path, "wb") as f:
        pickle.dump({"scaler": feat_scaler, "columns": norm_cols}, f)
    with open(tgt_path, "wb") as f:
        pickle.dump(tgt_scaler, f)

    print(f"    ✓ Scalers saved → {scalers_dir}/")
    return train, val, test, feat_scaler, tgt_scaler, norm_cols


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CITY PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_city(filepath, city_name, coords, output_dir):
    print(f"\n{'='*60}")
    print(f"  {city_name}")
    print(f"{'='*60}")

    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} cols")

    # ── Select columns ───────────────────────────────────────────────────────
    available   = [c for c in DIRECT_FEATURES if c in df.columns]
    missing_src = [c for c in DIRECT_FEATURES if c not in df.columns]
    if missing_src:
        print(f"  ⚠ Source columns not found (skipped): {missing_src}")

    keep = ["time", "is_day", "condition"] + available
    keep = [c for c in keep if c in df.columns]
    df   = df[keep].copy()

    # ── Add latent features ──────────────────────────────────────────────────
    df = add_latent_features(df)

    # ── Add lag features ─────────────────────────────────────────────────────
    df = add_lag_features(df)

    # ── Add coordinates ──────────────────────────────────────────────────────
    df["latitude"]  = coords["lat"]
    df["longitude"] = coords["lon"]
    df["city"]      = city_name

    # Drop raw string helpers (already encoded)
    df.drop(columns=[c for c in ["condition", "time"] if c in df.columns],
            inplace=True)

    full_report = {}

    # ── STEP 1: Sentinel values ──────────────────────────────────────────────
    print("\n  [1] Sentinel value check")
    df, sent_report = check_sentinel_values(df, city_name)
    full_report["sentinel"] = sent_report

    # ── STEP 2: Temporal continuity ──────────────────────────────────────────
    print("\n  [2] Temporal continuity check")
    df, time_report = check_temporal_continuity(df, city_name)
    full_report["temporal"] = time_report

    # ── STEP 3: NaN fill ─────────────────────────────────────────────────────
    print("\n  [3] NaN check & fill")
    df, nan_report = check_and_fill_missing(df, city_name)
    full_report["nan"] = nan_report

    # ── STEP 4: Split BEFORE normalization ───────────────────────────────────
    print("\n  [4] Train / val / test split")
    train, val, test = temporal_split(df)

    # ── STEP 5: Normalize ────────────────────────────────────────────────────
    print("\n  [5] Normalization")
    train, val, test, feat_sc, tgt_sc, norm_cols = normalize_splits(
        train, val, test, city_name, output_dir
    )

    print(f"\n  Final shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return train, val, test, full_report


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main(input_dir=".", output_dir="./processed"):
    os.makedirs(output_dir, exist_ok=True)

    all_reports = {}
    all_train, all_val, all_test = [], [], []

    for city, coords in CITIES.items():
        candidates = [
            os.path.join(input_dir, f"{city.lower()}_hourly_weather_5years.csv"),
            os.path.join(input_dir, f"{city}_hourly_weather_5years.csv"),
            os.path.join(input_dir, f"{city}.csv"),
            os.path.join(input_dir, f"{city.lower()}.csv"),
            os.path.join(input_dir, f"{city}_weather.csv"),
        ]
        filepath = next((p for p in candidates if os.path.exists(p)), None)
        if filepath is None:
            print(f"\n  ⚠ File not found for {city}, skipping.")
            continue

        train, val, test, report = process_city(
            filepath, city, coords, output_dir
        )
        all_reports[city] = report

        # Save per-city splits
        city_dir = os.path.join(output_dir, city)
        os.makedirs(city_dir, exist_ok=True)
        train.to_csv(os.path.join(city_dir, "train.csv"), index=False)
        val.to_csv(  os.path.join(city_dir, "val.csv"),   index=False)
        test.to_csv( os.path.join(city_dir, "test.csv"),  index=False)
        print(f"\n  ✓ Saved splits → {city_dir}/")

        all_train.append(train)
        all_val.append(val)
        all_test.append(test)

    # ── Combined datasets across all cities ──────────────────────────────────
    if all_train:
        combined_train = pd.concat(all_train, ignore_index=True)
        combined_val   = pd.concat(all_val,   ignore_index=True)
        combined_test  = pd.concat(all_test,  ignore_index=True)

        # Нормалізуємо lat/lon глобально (fit на train)
        from sklearn.preprocessing import StandardScaler as SS
        coord_scaler = SS()
        combined_train[["latitude","longitude"]] = coord_scaler.fit_transform(
            combined_train[["latitude","longitude"]])
        combined_val[["latitude","longitude"]]   = coord_scaler.transform(
            combined_val[["latitude","longitude"]])
        combined_test[["latitude","longitude"]]  = coord_scaler.transform(
            combined_test[["latitude","longitude"]])

        # Зберігаємо coord scaler
        scalers_dir = os.path.join(output_dir, "scalers")
        os.makedirs(scalers_dir, exist_ok=True)
        with open(os.path.join(scalers_dir, "coord_scaler.pkl"), "wb") as f:
            pickle.dump(coord_scaler, f)

        combined_train.to_csv(
            os.path.join(output_dir, "all_cities_train.csv"), index=False)
        combined_val.to_csv(
            os.path.join(output_dir, "all_cities_val.csv"),   index=False)
        combined_test.to_csv(
            os.path.join(output_dir, "all_cities_test.csv"),  index=False)
        print(f"\n✓ Combined train/val/test saved → {output_dir}/")

    # ── Full report ───────────────────────────────────────────────────────────
    report_path = os.path.join(output_dir, "preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"✓ Full report → {report_path}")

    # ── Feature list ──────────────────────────────────────────────────────────
    if all_train:
        sample = all_train[0]
        print("\n📋 Final feature columns:")
        for i, col in enumerate(sample.columns.tolist(), 1):
            tag = " ← TARGET (log1p + scaled)" if col == "precip_mm" else ""
            print(f"  {i:2}. {col}{tag}")


if __name__ == "__main__":
    import sys
    input_dir  = sys.argv[1] if len(sys.argv) > 1 else "."
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./processed"
    main(input_dir, output_dir)
