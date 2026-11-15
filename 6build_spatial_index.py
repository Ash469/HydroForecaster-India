import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

CATALOG_PATH = r"D:\GWT\dataset\stations_master.parquet"
OUTPUT_PARQUET = r"D:\GWT\dataset\spatial_distance_matrix.parquet"
OUTPUT_CSV = r"D:\GWT\dataset\spatial_distance_matrix.csv"

EARTH_RADIUS_KM = 6371.0  # Earth radius in kilometers

def latlon_to_cartesian(lat_deg, lon_deg):
    """Convert Latitude and Longitude in degrees to 3D Cartesian coordinates (x, y, z) on Earth sphere."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    
    x = EARTH_RADIUS_KM * np.cos(lat_rad) * np.cos(lon_rad)
    y = EARTH_RADIUS_KM * np.cos(lat_rad) * np.sin(lon_rad)
    z = EARTH_RADIUS_KM * np.sin(lat_rad)
    
    return np.column_stack((x, y, z))

def build_spatial_matrix():
    print("Starting Step 2.4: Spatial Distance Indexing (KD-Tree & Haversine Distance)...")

    if not os.path.exists(CATALOG_PATH):
        print(f"Error: {CATALOG_PATH} not found.")
        return

    df = pd.read_parquet(CATALOG_PATH)
    
    # Filter valid coordinates
    df_valid = df.dropna(subset=["latitude", "longitude"]).copy()
    print(f"Loaded {len(df_valid):,} valid station coordinates.")

    # Extract GWL stations
    gwl_df = df_valid[df_valid["category"] == "gwl"].reset_index(drop=True)
    print(f"Groundwater (GWL) Wells: {len(gwl_df):,}")

    if gwl_df.empty:
        print("No Groundwater wells found!")
        return

    gwl_coords_3d = latlon_to_cartesian(gwl_df["latitude"].values, gwl_df["longitude"].values)

    results_df = pd.DataFrame({
        "gwl_station_id": gwl_df["station_id"],
        "gwl_station_name": gwl_df["station_name"],
        "state": gwl_df["state"],
        "district": gwl_df["district"],
        "latitude": gwl_df["latitude"],
        "longitude": gwl_df["longitude"],
        "elevation_msl": gwl_df["elevation_msl"]
    })

    climate_categories = ["rainfall", "temperature", "humidity", "solar_radiation", "rwl", "pressure"]

    for cat in climate_categories:
        cat_df = df_valid[df_valid["category"] == cat].reset_index(drop=True)
        if cat_df.empty:
            print(f"  ⚠️ Skipping {cat.upper()}: No stations found.")
            continue

        cat_coords_3d = latlon_to_cartesian(cat_df["latitude"].values, cat_df["longitude"].values)
        tree = cKDTree(cat_coords_3d)

        # For rainfall, find top 3 nearest stations; for others, find top 1 nearest
        k_neighbors = 3 if cat == "rainfall" else 1
        distances, indices = tree.query(gwl_coords_3d, k=k_neighbors)

        if k_neighbors == 1:
            results_df[f"nearest_{cat}_id"] = cat_df.iloc[indices]["station_id"].values
            results_df[f"nearest_{cat}_name"] = cat_df.iloc[indices]["station_name"].values
            results_df[f"dist_km_{cat}"] = np.round(distances, 2)
        else:
            for i in range(k_neighbors):
                neighbor_idx = indices[:, i]
                neighbor_dist = distances[:, i]
                results_df[f"nearest_{cat}_{i+1}_id"] = cat_df.iloc[neighbor_idx]["station_id"].values
                results_df[f"nearest_{cat}_{i+1}_name"] = cat_df.iloc[neighbor_idx]["station_name"].values
                results_df[f"dist_km_{cat}_{i+1}"] = np.round(neighbor_dist, 2)

        avg_dist = np.mean(distances[:, 0]) if k_neighbors > 1 else np.mean(distances)
        print(f"  ✓ Linked {cat.upper()} ({len(cat_df)} stations) -> Avg Distance to GW Well: {avg_dist:.2f} km")

    # Save outputs
    results_df.to_parquet(OUTPUT_PARQUET, index=False)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSpatial Distance Matrix Construction Complete!")
    print(f"Total GW Wells Indexed: {len(results_df):,}")
    print(f"Saved distance matrix files:")
    print(f"  ✓ Parquet: {OUTPUT_PARQUET}")
    print(f"  ✓ CSV: {OUTPUT_CSV}")

    # Distance Distribution Summary
    print("\n=== SPATIAL PROXIMITY SUMMARY (Average Distance in KM) ===")
    dist_cols = [c for c in results_df.columns if c.startswith("dist_km_")]
    summary_dist = results_df[dist_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    summary_dist.columns = ["Mean (km)", "Std (km)", "Min (km)", "Median (km)", "Max (km)"]
    print(summary_dist.to_string())

if __name__ == "__main__":
    build_spatial_matrix()
