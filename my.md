Step 1---> unify_clean_csv.py
Drop the redundant columns from each csv and convert then to a standard csv.

Step 2---> 2convert_to_parquet.py 
Convert the cleaned csv files to parquet in load and do operations faster.

Step 3---> 3verify_parquet.py
Just for checking parquet

Step 4---> Cleaning of Parquets 
this is done in 03_PanIndia_Data_Cleaning_Pipeline.ipynb 

Step 5---> 5build_station_catalog.py
this combine all unique the station infomation from each feature (gwl,rainfall,temp etc).

Step 6 ---> 
this give about no of stations of each category in each state. like TamilNadu (gwl-690, rainfall-55, temp-9 etc).

Step 7 --->  6build_spatial_index.py



NOTEBOOKS;--->
01_PanIndia_Spatial_EDA.ipynb --> basic eda geting to how much data we have.

02_Full_dataInspection.ipynb --> full data inspection to know about outliers.

03_PanIndia_Data_Cleaning_Pipeline.ipynb --> full data cleaning 
reducing dataset from 52M to 4.3M remove hourly reading with day

04_PanIndia_Feature_Engineering.ipynb --> feature engineering 
