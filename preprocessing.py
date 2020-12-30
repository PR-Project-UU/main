import pandas as pd
import glob

# Allows for printing all columns; debug utility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Select only the data files in the data folder
data_files = glob.glob('data/*data.csv')

dfs = []
for file in data_files:
    dfs.append(pd.read_csv(file, encoding="ISO-8859-1"))

print(dfs[0])
print(dfs[1])
