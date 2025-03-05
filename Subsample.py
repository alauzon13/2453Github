import pandas as pd 
import random

random.seed(1013)

dfs = pd.read_excel("MasterTable_AI_FlowCAM.xlsx", sheet_name="MasterTable")

# Master table is unique for tiff files. So I'm selecting unique tiff files. 
subsample = dfs.sample(50)

def determine_odlocation(csvfile):
    if not csvfile.endswith(".csv"):
        csvfile += ".csv"
    return f"CSVs/{csvfile}"

# Apply the function to create the new column
subsample["ODLocation"] = subsample["csvfile"].apply(determine_odlocation)



subsample.to_csv("subsample.csv", index=False)






