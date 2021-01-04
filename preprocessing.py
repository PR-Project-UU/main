import pandas as pd
import glob

# Allows for printing all columns and increases width; debug utility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Select only the data files in the data folder
#data_files = glob.glob('data/*data.csv')
data_files = sorted(glob.glob('data/*data.csv'))

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

# The following allows to see all values of METROREG without doublets, that's how I retrieved the rows which we do not want
#sorted(set(list(filtered_data.METROREG)))

non_metro=['Non-metropolitan regions in Austria',
 'Non-metropolitan regions in Belgium',
 'Non-metropolitan regions in Bulgaria',
 'Non-metropolitan regions in Croatia',
 'Non-metropolitan regions in Czech Republic',
 'Non-metropolitan regions in Denmark',
 'Non-metropolitan regions in Estonia',
 'Non-metropolitan regions in Finland',
 'Non-metropolitan regions in Germany',
 'Non-metropolitan regions in Greece',
 'Non-metropolitan regions in Hungary',
 'Non-metropolitan regions in Ireland',
 'Non-metropolitan regions in Italy',
 'Non-metropolitan regions in Latvia',
 'Non-metropolitan regions in Lithuania',
 'Non-metropolitan regions in Malta',
 'Non-metropolitan regions in Netherlands',
 'Non-metropolitan regions in North Macedonia',
 'Non-metropolitan regions in Portugal',
 'Non-metropolitan regions in Romania',
 'Non-metropolitan regions in Serbia',
 'Non-metropolitan regions in Slovakia',
 'Non-metropolitan regions in Slovenia',
 'Non-metropolitan regions in Spain',
 'Non-metropolitan regions in Sweden',
 'Non-metropolitan regions in United Kingdom']
countries = ["West Midlands urban area","North Macedonia",
"Austria","Belgium","Bulgaria",
"Croatia","Cyprus","Czechia","Denmark",
"Estonia","Finland","France","Germany","Germany (until 1990 former territory of the FRG)"
,"Greece","Hungary","Ireland","Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands",
"Poland","Portugal","Romania","Serbia","Slovakia","Slovenia","Spain","Sweden"]

countries.extend(non_metro) #merges the two lists above

#Removing countries and zones we do not want
filtered_data2=filtered_data
for i in countries:
    filtered_data2=filtered_data2[filtered_data2.METROREG!=i]

#Preprocessing on the population column
filtered_data2.population = filtered_data2.population.str.replace(r':','0') #replacing missing values (noted ':') by 0
filtered_data2.population = filtered_data2['population'].str.replace(r',', '') #removing commas 
filtered_data2.population = filtered_data2['population'].astype(int) #changing column type

#Preprocessing on the GDP column
filtered_data2.gdp = filtered_data2.gdp.str.replace(r':','0') #replacing missing values (noted ':') by 0
filtered_data2.gdp = filtered_data2['gdp'].str.replace(r',', '') #removing commas 
filtered_data2.gdp = filtered_data2['gdp'].astype(float) #changing column type

#OPTIMAL VALUES TO BE DETERMINED
#As of now, we look at cities with minimum thresholds of population>500000 and GDP>40000, could be changed if we need more/less data
filtered_data3 = filtered_data2[(filtered_data2['TIME']>=2000) & (filtered_data2['population']>500000) & (filtered_data2['gdp']>40000)]

#Geolocation part

"""
To bridge the images and the socio economic data, we can use the geopy package to retrieve the latitude & longitude of the cities 
in our data set. From there, we need to figure out a way to access the satellite images w/latitude & longitude
"""
#Needed dependency:

#pip install geopy

import geopy
from geopy.geocoders import Nominatim
#quick example, to be removed
geolocator = Nominatim(user_agent="Master-Student") #I think we can put about anything in user agent it doesn't matter
location = geolocator.geocode("Paris")
print(location.latitude)

#Creating a function to retrieve latitude & longitude from a string

geolocator = Nominatim(user_agent="Master-Student")

def get_coord(k):
    assert type(k)==str
    return (geolocator.geocode(k).latitude,geolocator.geocode(k).longitude) # returns a tuple in the form (latitude,longitude)

#Next, we need to add an extra column in which we store the latitudes & longitudes of the corresponding rows
filtered_data3["latitude&longitude"]=filtered_data3["METROREG"].apply(get_coord)

"""This last part may ran into some trouble as some METROREG values 
from the last obtained dataframe 
are " Braunschweig-Salzgitter-Wolfsburg" ( can be checked with sorted(set(list(filtered_data3.METROREG)) ) 
needs a little bit of work to get it right but should work"""
