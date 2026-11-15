import os
import glob
import pandas as pd

BASE_CLEAN_DIR = r"D:\GWT\dataset\cleaned_csv"
OUTPUT_PARQUET_DIR = r"D:\GWT\dataset\parquet"

CATEGORIES = [
    "gwl",
    "rainfall",
    "temperature",
    "humidity",
    "solar_radiation",
    "rwl",
    "pressure"
]

def convert_all():
    print("Starting CSV to Parquet Compression Pipeline...")
    os.makedirs(OUTPUT_PARQUET_DIR, exist_ok=True)

    total_files = 0
    total_csv_size_mb = 0
    total_parquet_size_mb = 0

    for cat in CATEGORIES:
        in_folder = os.path.join(BASE_CLEAN_DIR, cat)
        if not os.path.exists(in_folder):
            print(f"Skipping missing category directory: {in_folder}")
            continue

        out_folder = os.path.join(OUTPUT_PARQUET_DIR, cat)
        os.makedirs(out_folder, exist_ok=True)

        csv_files = glob.glob(os.path.join(in_folder, "*.csv"))
        print(f"\n[Compressing Category: {cat.upper()}] ({len(csv_files)} files)")

        for csv_path in csv_files:
            fname = os.path.basename(csv_path)
            pq_fname = fname.replace(".csv", ".parquet")
            pq_path = os.path.join(out_folder, pq_fname)

            try:
                # Read CSV
                df = pd.read_csv(csv_path, low_memory=False)

                # Type optimizations for Parquet efficiency
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')

                float_cols = ["latitude", "longitude", "elevation_msl", "state_lgd_code", "district_lgd_code"]
                # Add measurement value column
                val_cols = [c for c in df.columns if c not in ["station_name", "agency", "state", "district", "basin", "river", "timestamp"]]
                for col in float_cols + val_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype("float32")

                # Convert string metadata to efficient categorical / string types
                string_cols = ["station_name", "agency", "state", "district", "basin", "river"]
                for col in string_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna("").astype("string")

                # Write Parquet with snappy compression
                df.to_parquet(pq_path, engine="pyarrow", compression="snappy", index=False)

                csv_sz = os.path.getsize(csv_path) / (1024 * 1024)
                pq_sz = os.path.getsize(pq_path) / (1024 * 1024)

                total_csv_size_mb += csv_sz
                total_parquet_size_mb += pq_sz
                total_files += 1

                print(f"  ✓ {pq_fname}: {csv_sz:.1f} MB (CSV) → {pq_sz:.1f} MB (Parquet) [{((1 - pq_sz/csv_sz)*100):.1f}% smaller]")

            except Exception as e:
                print(f"  ❌ Error converting {fname}: {e}")

    print(f"\nCompression Complete!")
    print(f"Total Files Converted: {total_files}")
    print(f"Total Disk Space: {total_csv_size_mb:.2f} MB (CSV) → {total_parquet_size_mb:.2f} MB (Parquet)")
    if total_csv_size_mb > 0:
        overall_saving = (1 - total_parquet_size_mb / total_csv_size_mb) * 100
        print(f"Overall Storage Reduction: {overall_saving:.1f}%")
    print(f"All Parquet files saved in: {OUTPUT_PARQUET_DIR}")

if __name__ == "__main__":
    convert_all()
