"""
Pipeline Step 4: Export Latest Models & Dashboard Stores Runner
===============================================================
Executes notebooks/10_Model_Serialization_and_Dashboard_Export.ipynb in-place.
Takes the latest trained models and feature matrix and exports:
  1. Serialized inference models (models/ridge_t_*.joblib, models/rf_t_*.joblib, models/lgb_t_*.txt)
  2. Standardized feature vector (models/feature_names.joblib)
  3. Lightweight Parquet stores in dashboard_data/ (stations_directory, monthly_station_features, recent_telemetry)
"""

import os
import sys
import time
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
NB_NAME = "10_Model_Serialization_and_Dashboard_Export.ipynb"

def main():
    nb_path = os.path.join(NOTEBOOKS_DIR, NB_NAME)
    if not os.path.exists(nb_path):
        print(f"Error: {NB_NAME} not found at {nb_path}")
        sys.exit(1)

    print("\n" + "#" * 80)
    print("  STEP 4: MODEL SERIALIZATION & DASHBOARD EXPORT (NOTEBOOK 10)")
    print("#" * 80)

    start_t = time.time()
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=1800",
        nb_path
    ]

    res = subprocess.run(cmd, cwd=NOTEBOOKS_DIR)
    elapsed = time.time() - start_t

    if res.returncode == 0:
        print("\n" + "=" * 80)
        print(f"  SUCCESS: Latest models & dashboard stores exported in {elapsed:.1f}s!")
        print("  Streamlit dashboard is now completely up-to-date.")
        print("=" * 80 + "\n")
    else:
        print(f"\n  FAILED: Notebook 10 export exited with code {res.returncode}!")
        sys.exit(1)

if __name__ == "__main__":
    main()

