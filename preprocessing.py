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

non_metro=['Non-metropolitan regions in Austria','Non-metropolitan regions in Belgium','Non-metropolitan regions in Bulgaria','Non-metropolitan regions in Croatia',
 'Non-metropolitan regions in Czech Republic','Non-metropolitan regions in Denmark','Non-metropolitan regions in Estonia','Non-metropolitan regions in Finland',
 'Non-metropolitan regions in Germany','Non-metropolitan regions in Greece','Non-metropolitan regions in Hungary','Non-metropolitan regions in Ireland',
 'Non-metropolitan regions in Italy','Non-metropolitan regions in Latvia','Non-metropolitan regions in Lithuania','Non-metropolitan regions in Malta',
 'Non-metropolitan regions in Netherlands','Non-metropolitan regions in North Macedonia','Non-metropolitan regions in Portugal','Non-metropolitan regions in Romania',
 'Non-metropolitan regions in Serbia','Non-metropolitan regions in Slovakia','Non-metropolitan regions in Slovenia','Non-metropolitan regions in Spain',
 'Non-metropolitan regions in Sweden','Non-metropolitan regions in United Kingdom']

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

#Creating a function to retrieve latitude & longitude from a string

geolocator = Nominatim(user_agent="Master-Student")

def get_coord(k):
    assert type(k) == str
    return (geolocator.geocode(k).latitude,geolocator.geocode(k).longitude) # returns a tuple in the form (latitude,longitude)

#Next, we need to add an extra column in which we store the latitudes & longitudes of the corresponding rows

latlong = [0 for i in range(filtered_data3.shape[0])] #list full of zeros, could improve the code w/numpy later

#Adding the extra  empty column:
filtered_data3['latlong']=latlong

d={} #Creating a dictionnary with cities as keys and position as values
for i in sorted(set(list(filtered_data3.METROREG))):
    d[i]=get_coord(i)

#Adding corresponding positions to cities
#Problem : was only able to add the coordinates as strings and not tuples :(, this needs to be changed otherwise it's gonna slow computation speed
for i in d.keys():
    for idx, row in filtered_data3.iterrows():
        if  filtered_data3.loc[idx,'METROREG'] == i:
            filtered_data3.loc[idx,'latlong'] = str(d[i])
#We need to delete the cities for which we do not have enough data for, here I set the the minimum number of instances to be at least 10 years of data per city

d2=dict(filtered_data3['METROREG'].value_counts()) #This dictionnary stores the number of times each city appears in the data


for i in d2.keys(): #This for loop removes all rows for cities on which we do not have enough data
    if d2[i]<10: #10 years at least
        filtered_data3=filtered_data3[filtered_data3.METROREG!=i]
#If everything worked correctly, you should be left with the filtered_data3 dataframe in the shape (633,6)
