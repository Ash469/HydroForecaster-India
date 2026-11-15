import os
import glob
import pandas as pd

PARQUET_DIR = r"D:\GWT\dataset\parquet"

categories = ["gwl", "rainfall", "temperature", "humidity", "solar_radiation", "rwl", "pressure"]

print("=== PARQUET INTEGRITY & VERIFICATION REPORT ===")

total_rows_all = 0
total_files_all = 0

for cat in categories:
    folder = os.path.join(PARQUET_DIR, cat)
    files = glob.glob(os.path.join(folder, "*.parquet"))
    total_files_all += len(files)
    
    cat_rows = 0
    sample_file = files[0] if files else None
    
    if sample_file:
        df_sample = pd.read_parquet(sample_file)
        cols = list(df_sample.columns)
        dtypes = {col: str(dtype) for col, dtype in df_sample.dtypes.items()}
        
        # Calculate total rows across all files in category
        for f in files:
            df_f = pd.read_parquet(f, columns=[cols[0]])
            cat_rows += len(df_f)
            
        total_rows_all += cat_rows
        
        print(f"\n📁 Category: {cat.upper()} ({len(files)} files, {cat_rows:,} total rows)")
        print(f"   Sample File: {os.path.basename(sample_file)}")
        print(f"   Columns & Types: {dtypes}")
        print(f"   Date Range: {df_sample['timestamp'].min()} to {df_sample['timestamp'].max()}")
        print(f"   Sample Data Row 1: {df_sample.iloc[0].to_dict()}")

print(f"\n==============================================")
print(f"TOTAL PARQUET FILES: {total_files_all}")
print(f"TOTAL OBSERVATION ROWS ACROSS ALL INDIA: {total_rows_all:,}")
print(f"==============================================")
