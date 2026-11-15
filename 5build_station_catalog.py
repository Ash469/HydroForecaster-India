import os
import glob
import pandas as pd
import numpy as np

PARQUET_BASE_DIR = r"D:\GWT\dataset\parquet"
OUTPUT_DIR = r"D:\GWT\dataset"

CATEGORIES = [
    "gwl",
    "rainfall",
    "temperature",
    "humidity",
    "solar_radiation",
    "rwl",
    "pressure"
]

def build_catalog():
    print("Starting Master Station Inventory Catalog Construction...")
    
    station_records = []

    for cat in CATEGORIES:
        folder = os.path.join(PARQUET_BASE_DIR, cat)
        if not os.path.exists(folder):
            print(f"Directory not found: {folder}")
            continue

        files = glob.glob(os.path.join(folder, "*.parquet"))
        print(f"\n[Scanning Category: {cat.upper()}] ({len(files)} files)")

        for f in files:
            try:
                df = pd.read_parquet(f)
                val_col = [c for c in df.columns if c not in [
                    "station_name", "agency", "state_lgd_code", "state",
                    "district_lgd_code", "district", "basin", "river",
                    "latitude", "longitude", "elevation_msl", "timestamp"
                ]][0]

                # Group by station_name to extract station-level metadata & metrics
                grouped = df.groupby("station_name")

                for st_name, group in grouped:
                    st_name_clean = str(st_name).strip()
                    if not st_name_clean:
                        continue

                    first_row = group.iloc[0]
                    state_name = str(first_row.get("state", "")).strip()
                    district_name = str(first_row.get("district", "")).strip()
                    agency_name = str(first_row.get("agency", "")).strip()
                    basin_name = str(first_row.get("basin", "")).strip()
                    river_name = str(first_row.get("river", "")).strip()

                    lat = group["latitude"].dropna().iloc[0] if not group["latitude"].dropna().empty else np.nan
                    lon = group["longitude"].dropna().iloc[0] if not group["longitude"].dropna().empty else np.nan
                    elev = group["elevation_msl"].dropna().iloc[0] if not group["elevation_msl"].dropna().empty else np.nan

                    state_code = first_row.get("state_lgd_code", np.nan)
                    district_code = first_row.get("district_lgd_code", np.nan)

                    total_recs = len(group)
                    missing_val_cnt = group[val_col].isna().sum()
                    missing_pct = round((missing_val_cnt / total_recs) * 100, 2) if total_recs > 0 else 0.0

                    valid_times = group["timestamp"].dropna()
                    start_date = valid_times.min() if not valid_times.empty else None
                    end_date = valid_times.max() if not valid_times.empty else None

                    # Generate clean Station ID
                    cat_prefix = cat.upper()
                    st_id = f"{cat_prefix}_{state_name[:3].upper()}_{st_name_clean.replace(' ', '_')}"

                    station_records.append({
                        "station_id": st_id,
                        "category": cat,
                        "station_name": st_name_clean,
                        "agency": agency_name,
                        "state_lgd_code": state_code,
                        "state": state_name,
                        "district_lgd_code": district_code,
                        "district": district_name,
                        "basin": basin_name,
                        "river": river_name,
                        "latitude": lat,
                        "longitude": lon,
                        "elevation_msl": elev,
                        "total_records": total_recs,
                        "missing_value_count": missing_val_cnt,
                        "missing_percent": missing_pct,
                        "start_date": start_date,
                        "end_date": end_date
                    })

            except Exception as e:
                print(f"Error processing {os.path.basename(f)}: {e}")

    catalog_df = pd.DataFrame(station_records)
    
    # Save Catalog
    csv_path = os.path.join(OUTPUT_DIR, "stations_master.csv")
    parquet_path = os.path.join(OUTPUT_DIR, "stations_master.parquet")

    catalog_df.to_csv(csv_path, index=False)
    catalog_df.to_parquet(parquet_path, index=False)

    print(f"\nMaster Station Inventory Construction Complete!")
    print(f"Total Unique Stations Cataloged Across India: {len(catalog_df):,}")
    print(f"Saved catalog files:")
    print(f"  ✓ CSV: {csv_path}")
    print(f"  ✓ Parquet: {parquet_path}")

    # Summary by Category
    print("\n--- STATION COUNT BREAKDOWN BY CATEGORY ---")
    summary = catalog_df.groupby("category").agg(
        total_stations=("station_id", "count"),
        avg_records_per_station=("total_records", lambda x: int(x.mean())),
        avg_missing_percent=("missing_percent", lambda x: round(x.mean(), 2))
    ).reset_index()
    print(summary.to_string(index=False))

if __name__ == "__main__":
    build_catalog()
