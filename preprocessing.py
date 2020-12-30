import pandas as pd
import glob

# Allows for printing all columns and increases width; debug utility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Select only the data files in the data folder
data_files = glob.glob('data/*data.csv')

# Create the dfs and merge them on the keys TIME and METROREG
merged_data = None
for file in data_files:
    current_df = pd.read_csv(file, encoding="ISO-8859-1")
    if merged_data is not None:
        merged_data = pd.merge(merged_data, current_df, on=["TIME", "METROREG"])
    else:
        merged_data = current_df

# Rename relevant columns and drop the useless ones
# Though, be aware that measurement information is also lost
renames = {'Value_x': 'employed_persons',
           'Value_y': 'gdp',
           'Value': 'population'}

droppes = ['UNIT_x', 'WSTATUS', 'NACE_R2', 'Flag and Footnotes_x',
           'UNIT_y', 'Flag and Footnotes_y', 'AGE', 'SEX', 'Flag and Footnotes']

filtered_data = merged_data.rename(columns=renames).drop(columns=droppes)

#print(filtered_data)
