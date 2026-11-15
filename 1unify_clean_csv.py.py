import os
import glob
import pandas as pd
import numpy as np

BASE_DIR = r"D:\GWT\dataset\raw"
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "cleaned_csv")

# Folder to category and measurement column mapping
FOLDER_CONFIG = {
    "GWL state (2021-2025)": {
        "out_dir": "gwl",
        "value_col_name": "gwl_m",
        "value_match_keywords": ["groundwater level", "gwl", "ground water level"]
    },
    "rainfall state (2021-2025)": {
        "out_dir": "rainfall",
        "value_col_name": "rainfall_mm",
        "value_match_keywords": ["rainfall", "rain"]
    },
    "temp state (2021-2025)": {
        "out_dir": "temperature",
        "value_col_name": "temp_c",
        "value_match_keywords": ["temperature", "temp"]
    },
    "rel humidity state (2021-2025)": {
        "out_dir": "humidity",
        "value_col_name": "humidity_pct",
        "value_match_keywords": ["relative humidity", "humidity", "humid"]
    },
    "solar radiation state (2021-2025)": {
        "out_dir": "solar_radiation",
        "value_col_name": "solar_wm2",
        "value_match_keywords": ["solar radiation", "solar"]
    },
    "RWL state (2021-2025)": {
        "out_dir": "rwl",
        "value_col_name": "rwl_m",
        "value_match_keywords": ["river water level", "rwl"]
    },
    "atomspheric pressure (2021-2025)": {
        "out_dir": "pressure",
        "value_col_name": "pressure_mb",
        "value_match_keywords": ["atmospheric pressure", "pressure"]
    }
}

# Standard Output Column Order
STANDARD_COLUMNS = [
    "station_name",
    "agency",
    "state_lgd_code",
    "state",
    "district_lgd_code",
    "district",
    "basin",
    "river",
    "latitude",
    "longitude",
    "elevation_msl",
    "timestamp"
]

def find_column(cols, keywords, exclude_keywords=[]):
    """Find column matching any of the keywords (preferring exact match, excluding unwanted terms)."""
    # 1. Exact match pass
    for col in cols:
        col_clean = str(col).strip().lower()
        if any(col_clean == kw.lower() for kw in keywords):
            return col

    # 2. Substring match pass (excluding unwanted terms like 'lgd', 'code')
    for col in cols:
        col_lower = str(col).lower()
        if any(ex in col_lower for ex in exclude_keywords):
            continue
        for kw in keywords:
            if kw in col_lower:
                return col
    return None

def clean_file(file_path, config):
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    cols = df.columns

    # Standard column matchers
    station_col = find_column(cols, ["station"])
    agency_col = find_column(cols, ["agency"])
    state_lgd_col = find_column(cols, ["state lgd code", "state lgd"])
    state_col = find_column(cols, ["state"], exclude_keywords=["lgd", "code"])
    district_lgd_col = find_column(cols, ["district lgd code", "district lgd"])
    district_col = find_column(cols, ["district"], exclude_keywords=["lgd", "code"])
    basin_col = find_column(cols, ["basin"])
    river_col = find_column(cols, ["river"])
    lat_col = find_column(cols, ["latitude", "lat"])
    long_col = find_column(cols, ["longitude", "lon", "long"])
    
    # Elevation column variations
    elev_col = find_column(cols, ["rl_msl", "rl_of_zerogauge", "meansealevel", "msl"])
    
    # Time column
    time_col = find_column(cols, ["data acquisition time", "time", "date"])

    # Measurement column
    val_col = find_column(cols, config["value_match_keywords"])

    if not time_col or not val_col:
        print(f"Skipping {os.path.basename(file_path)}: missing time or value column")
        return None

    cleaned_df = pd.DataFrame()
    cleaned_df["station_name"] = df[station_col].astype(str).str.strip() if station_col else ""
    cleaned_df["agency"] = df[agency_col].astype(str).str.strip() if agency_col else ""
    
    # State & District (Names vs LGD Codes)
    cleaned_df["state_lgd_code"] = pd.to_numeric(df[state_lgd_col], errors='coerce') if state_lgd_col else np.nan
    cleaned_df["state"] = df[state_col].astype(str).str.strip() if state_col else ""
    cleaned_df["district_lgd_code"] = pd.to_numeric(df[district_lgd_col], errors='coerce') if district_lgd_col else np.nan
    cleaned_df["district"] = df[district_col].astype(str).str.strip() if district_col else ""

    cleaned_df["basin"] = df[basin_col].astype(str).str.strip() if basin_col else ""
    cleaned_df["river"] = df[river_col].astype(str).str.strip() if river_col else ""

    # Lat/Long/Elevation
    cleaned_df["latitude"] = pd.to_numeric(df[lat_col], errors='coerce') if lat_col else np.nan
    cleaned_df["longitude"] = pd.to_numeric(df[long_col], errors='coerce') if long_col else np.nan
    cleaned_df["elevation_msl"] = pd.to_numeric(df[elev_col], errors='coerce') if elev_col else np.nan

    # Timestamp standardization
    cleaned_df["timestamp"] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

    # Target Measurement Value
    val_col_target = config["value_col_name"]
    cleaned_df[val_col_target] = pd.to_numeric(df[val_col], errors='coerce')

    # Replace '-' string representations with empty NaN
    cleaned_df = cleaned_df.replace({"-": np.nan, "": np.nan})

    # Ensure column order
    final_cols = STANDARD_COLUMNS + [val_col_target]
    cleaned_df = cleaned_df[final_cols]

    return cleaned_df

def process_all():
    print("Starting CSV Cleaning and Schema Unification Pipeline (V2 with Exact Column Matching)...")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    total_processed = 0

    for folder_name, config in FOLDER_CONFIG.items():
        folder_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.exists(folder_path):
            print(f"Directory not found: {folder_path}")
            continue

        out_folder = os.path.join(OUTPUT_BASE_DIR, config["out_dir"])
        os.makedirs(out_folder, exist_ok=True)

        files = glob.glob(os.path.join(folder_path, "*.csv"))
        print(f"\n[Processing Category: {config['out_dir'].upper()}] ({len(files)} files)")

        for f in files:
            fname = os.path.basename(f)
            out_fname = f"clean_{fname}"
            out_path = os.path.join(out_folder, out_fname)

            cleaned_df = clean_file(f, config)
            if cleaned_df is not None:
                cleaned_df.to_csv(out_path, index=False)
                total_processed += 1
                print(f"  ✓ Cleaned & saved: {config['out_dir']}/{out_fname} ({len(cleaned_df):,} rows)")

    print(f"\nPipeline finished! Successfully processed {total_processed} files.")
    print(f"All cleaned CSVs are saved in: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    process_all()
