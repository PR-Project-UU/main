import pandas as pd
import glob
import numpy as np
import geopy
from geopy.geocoders import Nominatim
from functools import partial


# Allows for printing all columns and increases width; debug utility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Select only the data files in the data folder
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

# print(filtered_data)

# The following allows to see all values of METROREG without doublets, that's how I retrieved the rows which we
# do not want
# sorted(set(list(filtered_data.METROREG)))

non_metro = ['Non-metropolitan regions in Austria', 'Non-metropolitan regions in Belgium',
             'Non-metropolitan regions in Bulgaria', 'Non-metropolitan regions in Croatia',
             'Non-metropolitan regions in Czech Republic', 'Non-metropolitan regions in Denmark',
             'Non-metropolitan regions in Estonia', 'Non-metropolitan regions in Finland',
             'Non-metropolitan regions in Germany', 'Non-metropolitan regions in Greece',
             'Non-metropolitan regions in Hungary', 'Non-metropolitan regions in Ireland',
             'Non-metropolitan regions in Italy', 'Non-metropolitan regions in Latvia',
             'Non-metropolitan regions in Lithuania', 'Non-metropolitan regions in Malta',
             'Non-metropolitan regions in Netherlands', 'Non-metropolitan regions in North Macedonia',
             'Non-metropolitan regions in Portugal', 'Non-metropolitan regions in Romania',
             'Non-metropolitan regions in Serbia', 'Non-metropolitan regions in Slovakia',
             'Non-metropolitan regions in Slovenia', 'Non-metropolitan regions in Spain',
             'Non-metropolitan regions in Sweden', 'Non-metropolitan regions in United Kingdom']

countries = ["United Kingdom", "West Midlands urban area", "North Macedonia", "Austria", "Belgium", "Bulgaria",
             "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany",
             "Germany (until 1990 former territory of the FRG)", "Greece", "Hungary", "Ireland", "Italy",
             "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania",
             "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden"]

# Merges the two lists above
countries.extend(non_metro)

# Removing countries and zones we do not want
filtered_data2 = filtered_data
for i in countries:
    filtered_data2 = filtered_data2[filtered_data2.METROREG != i]

# Preprocessing on the population column
filtered_data2.population = filtered_data2.population.str.replace(r':', '0')  # replacing NAs (noted ':') by 0
filtered_data2.population = filtered_data2['population'].str.replace(r',', '')  # removing commas
filtered_data2.population = filtered_data2['population'].astype(int)  # changing column type

# Preprocessing on the GDP column
filtered_data2.gdp = filtered_data2.gdp.str.replace(r':', '0')  # replacing missing values (noted ':') by 0
filtered_data2.gdp = filtered_data2['gdp'].str.replace(r',', '')  # removing commas
filtered_data2.gdp = filtered_data2['gdp'].astype(float)  # changing column type

# Employed_persons column
filtered_data2.employed_persons = filtered_data2.employed_persons.astype(str).str.replace(r':', '0')
filtered_data2.employed_persons = filtered_data2['employed_persons'].str.replace(r',', '')
filtered_data2.employed_persons = filtered_data2['employed_persons'].astype(float)

# OPTIMAL VALUES TO BE DETERMINED
# As of now, we look at cities with minimum thresholds of population>500000 and GDP>40000,
# could be changed if we need more/less data
filtered_data3 = filtered_data2[(filtered_data2['TIME'] >= 2000) &
                                (filtered_data2['population'] > 500000) &
                                (filtered_data2['gdp'] > 40000)]

###### Geolocation part

"""
To bridge the images and the socioeconomic data, we can use the geopy package to retrieve the latitude & 
longitude of the cities in our data set. From there, we need to figure out a way to access the satellite 
images w/latitude & longitude
"""

# # Directly covered from https://developers.google.com/earth-engine/guides/landsat
# # Load a raw Landsat scene and display it.
# raw = ee.Image('LANDSAT/LC08/C01/T1/LC08_044034_20140318');
# Map.centerObject(raw, 10);
# Map.addLayer(raw, {bands: ['B4', 'B3', 'B2'], min: 6000, max: 12000}, 'raw');

# # Convert the raw data to radiance.
# var radiance = ee.Algorithms.Landsat.calibratedRadiance(raw);
# Map.addLayer(radiance, {bands: ['B4', 'B3', 'B2'], max: 90}, 'radiance');

# # Convert the raw data to top-of-atmosphere reflectance.
# toa = ee.Algorithms.Landsat.TOA(raw);
# Map.addLayer(toa, {bands: ['B4', 'B3', 'B2'], max: 0.2}, 'toa reflectance');

######

# Creating a function to retrieve latitude & longitude from a string
geo_locator = Nominatim(user_agent="Master-Student")


# returns a tuple in the form (latitude,longitude)
def get_coord(k):
    assert type(k) == str
    return geo_locator.geocode(k).latitude, geo_locator.geocode(k).longitude


# Next, we need to add an extra column in which we store the latitudes & longitudes of the corresponding rows
filtered_data3['latitude'] = filtered_data3['longitude'] = np.zeros(len(filtered_data3))

# Creating a list with cities and their coords
coords_list = []
for city in sorted(list(set(list(filtered_data3.METROREG))))[:5]:  # [:5] is voor debug, weghalen later
    try:
        city_coords = get_coord(city)
        coords_list.append([city, city_coords])
    except:
        print("Couldn't get coords of:", city)


# Adding corresponding positions to cities
for coords_city in coords_list:
    city = coords_city[0]
    coords = coords_city [1]
    filtered_data3.loc[filtered_data3['METROREG'] == city, 'latitude'] = coords[0]
    filtered_data3.loc[filtered_data3['METROREG'] == city, 'longitude'] = coords[1]

# We need to delete the cities for which we do not have enough data for,
# here I set the the minimum number of instances to be at least 10 years of data per city
# This dictionary stores the number of times each city appears in the data
d2 = dict(filtered_data3['METROREG'].value_counts())

# This for loop removes all rows for cities on which we do not have enough data
for i in d2.keys():
    if d2[i] < 10:  # 10 years at least
        # Yields filtered_data3 in the shape (614, 7)
        filtered_data3 = filtered_data3[filtered_data3.METROREG != i]

# Print(filtered_data3)
