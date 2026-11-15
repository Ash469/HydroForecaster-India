"""
Pipeline Step 1: Data Preprocessing & Spatial Indexing Runner
============================================================
Sequentially executes all initial data processing and indexing scripts:
  1. 1unify_clean_csv.py.py      -> Unifies irregular headers & cleans raw CSVs
  2. 2convert_to_parquet.py       -> Compresses cleaned CSVs to snappy Parquet
  3. 5build_station_catalog.py    -> Reconstructs dataset/stations_master.parquet
  4. 6build_spatial_index.py      -> Recomputes KD-Tree nearest neighbor distance matrix
"""

import os
import sys
import time
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def print_header(title):
    border = "=" * 80
    print(f"\n{border}")
    print(f"  >>> {title}")
    print(f"{border}\n")

def run_script(script_name):
    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"Error: Script {script_name} not found at {script_path}")
        return False

    print_header(f"RUNNING: {script_name}")
    start_t = time.time()
    res = subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
    elapsed = time.time() - start_t

    if res.returncode == 0:
        print(f"  SUCCESS: {script_name} completed in {elapsed:.1f}s.")
        return True
    else:
        print(f"  FAILED: {script_name} exited with code {res.returncode}!")
        return False

def main():
    start_total = time.time()
    print("\n" + "#" * 80)
    print("  STEP 1: RAW TELEMETRY PREPROCESSING & SPATIAL INDEXING PIPELINE")
    print("#" * 80)

    scripts = [
        "1unify_clean_csv.py.py",
        "2convert_to_parquet.py",
        "5build_station_catalog.py",
        "6build_spatial_index.py"
    ]

    for s in scripts:
        if not run_script(s):
            print(f"\nPreprocessing pipeline stopped due to error in {s}.")
            sys.exit(1)

    elapsed_total = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"  ALL PREPROCESSING SCRIPTS FINISHED IN {elapsed_total/60:.2f} MINUTES!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

