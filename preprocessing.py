import pandas as pd
import glob

# Allows for printing all columns and increases width; debug utility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Select only the data files in the data folder
data_files = glob.glob('data/*data.csv')

# We create the dfs and merge them on the keys TIME and METROREG
dfs = None
for file in data_files:
    current_df = pd.read_csv(file, encoding="ISO-8859-1")
    if dfs is not None:
        dfs = pd.merge(dfs, current_df, on=["TIME", "METROREG"])
    else:
        dfs = current_df

print(dfs)