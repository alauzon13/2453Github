# This script will select necessary tif files, based on data_cleaned. 
import pandas as pd 
import os
import shutil

# Read the list of TIFF files from data_cleaned.csv
data_cleaned = pd.read_csv("data_cleaned.csv")
tifs = data_cleaned["Image.File"]

# Specify the source directory (OneDrive) and the destination directory (tifs folder)
source_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/HURON_OverlapTiffsWithPP" 
destination_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/tifs"

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Iterate through the list of TIFF files and move each file
for tif in tifs:
    source_path = os.path.join(source_dir, tif)
    destination_path = os.path.join(destination_dir, tif)
    
    # Check if the file exists in the source directory before moving
    if os.path.exists(source_path):
        try:
            shutil.move(source_path, destination_path)
            print(f"Moved {tif} to {destination_dir}")
        except Exception as e:
            print(f"Error moving {tif}: {e}")
    else:
        print(f"File not found: {source_path}")

print("Finished moving TIFF files.")