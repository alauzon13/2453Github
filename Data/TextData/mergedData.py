# Create merged dataset
import pandas as pd 
import os

# Load data
subsample = pd.read_csv("subsample.csv")
master = pd.read_excel("MasterTable_AI_FlowCAM.xlsx", sheet_name="MasterTable")

# Set the base directory where the CSV files are located
base_dir = "/Users/adelelauzon/Desktop/MSc/STA5243"
# Extract necessary CSVs based on subsample
stacked_csvs = pd.DataFrame()

for csv_location in subsample["ODLocation"]:
    try:
        # Construct the full file path
        full_file_path = os.path.join(base_dir, csv_location)
        # Load the CSV file
        csv_to_add = pd.read_csv(full_file_path)
        # Add new column with the name of the CSV
        name_of_file = os.path.basename(full_file_path)
        csv_to_add["csvfile"] = name_of_file
        stacked_csvs = pd.concat([stacked_csvs, csv_to_add], ignore_index=True)
    except FileNotFoundError:
        print(f"File not found: {full_file_path}")
        continue

# Indicate that the script has finished searching through the list
print("Finished searching through the list of CSV files.")

#stacked_csvs.to_csv("stacked_csvs.csv", index=False)

stacked_csvs = pd.read_csv("stacked_csvs.csv")

# Merge the stacked CSVs with the master DataFrame on a common column (e.g., 'csvfile')
merged_data = pd.merge(stacked_csvs, master, left_on="Image.File", right_on="tifffile", how="left")

# Save the merged data to a new CSV file
merged_data.to_csv("merged_data.csv", index=False)

# Filter the merged data based on a list of values
values_to_filter = ["Calanoid_1", "Cylopoid_1", "Bosmina_1", "Herpacticoida", "Chironomid", 
                    "Chydoridae", "Daphnia"]
merged_data_filtered = merged_data[merged_data["Class"].isin(values_to_filter)]

# Filter to only include unique rows 
unique_merged_data_filtered = merged_data_filtered.drop_duplicates()

# Save the filtered data to a new CSV file
unique_merged_data_filtered.to_csv("filtered_data_new.csv", index=False)
