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

#Each city has a 19 instances of data : [2000,2018]
#code for the missing values
#I have not yet done the case where we have "holes" 
def treat_missing(k):
    #nested helper function
    #needed because .index() doesn't retrieve position for more than one value
    #eg: for l=[0,0,2], l.index(0) returns 0 and not (0,1)
    def get_zeros(k):
        z=[]
        for i in range(len(k)):
            if k[i]==0:
                z.append(i)
        return z  
    #turns the dataframe into lists
    L1 = list(k.employed_persons)
    L2 = list(k.gdp)
    L3 = list(k.population) 
    z1=get_zeros(L1)
    z2=get_zeros(L2)
    z3=get_zeros(L3)
    #for missing single values at either ends 
    
    #employed_persons
    if len(z1) == 1 and z1[0] == 0:
        L1[0] = L1[1] - abs(L1[1]-L1[2])
        k.employed_persons = L1
    if len(z1) == 1 and z1[0] == 18:
        L1[18] = L1[17] + abs(L1[17]-L1[16])
        k.employed_persons = L1
    
    #gdp
    if len(z2) == 1 and z2[0] == 0:
        L2[0] = L2[1] - abs(L2[1]-L2[2])
        k.gdp = L2
    if len(z2) == 1 and z2[0] == 18:
        L2[18] = L2[17] + abs(L2[17]-L2[16])
        k.gdp = L2
    
    #population
    if len(z3) == 1 and z3[0] == 0:
        L3[0] = L3[1] - abs(L3[1]-L3[2])
        k.population = L3
    if len(z3) == 1 and z3[0] == 18:
        L3[18] = L3[17] + abs(L3[17]-L3[16])
        k.population = L3
    #For missing values at ends (more than one)
    #nested function
    #[0,1]->True
    #[0,2]->False
    def evenly_spaced(l):
        for i in range(len(l)-1):
            if abs(l[i]-l[i+1])!=1:
                return False
        return True

    #employed_persons
    if (len(z1) >1) and evenly_spaced(z1) == True and z1[0] == 0:
        for i in reversed(z1):
            L1[i] = L1[i+1] - abs(L1[i+1] - L1[i+2])
        k.employed_persons = L1
    
    #gdp
    if (len(z2) >1) and evenly_spaced(z2) == True and z2[0] == 0:
        for i in reversed(z2):
            L2[i] = L2[i+1] - abs(L2[i+1] - L2[i+2])
        k.gdp = L2
        
    #population
    if (len(z3) >1) and evenly_spaced(z3) == True and z3[0] == 0:
        for i in reversed(z3):
            L3[i] = L3[i+1] - abs(L3[i+1] - L3[i+2])
        k.population = L3
        
    #other end 
    #employed_persons
    if (len(z1) >1) and evenly_spaced(z1) == True and z1[-1] == 18:
        for i in z1:
            L1[i] = L1[i-1] + abs(L1[i-1] - L1[i-2])
        k.employed_persons = L1
    #gdp
    if (len(z2) >1) and evenly_spaced(z2) == True and z2[-1] == 18:
        for i in z2:
            L2[i] = L2[i-1] + abs(L2[i-1] - L2[i-2])
        k.gdp = L2
    #population
    if (len(z3) >1) and evenly_spaced(z3) == True and z3[-1] == 18:
        for i in z3:
            L3[i] = L3[i-1] + abs(L3[i-1] - L3[i-2])
        k.population = L3
    #If values are missing in the middle of the dataframe: -> apply what Simon showed on excel
    #employed_persons
    if ((len(z1) >1) and evenly_spaced(z1) == True) and (z1[-1] != 18 and z1[0] != 0):
        p = L1[z1[0]-1] - L1[z1[-1]+1]
        add = p/(len(z1)+1)
        for i in z1:
            L1[i]=L1[i-1]+add
        k.employed_persons=L1
    #gdp
    if ((len(z2) >1) and evenly_spaced(z2) == True) and (z2[-1] != 18 and z2[0] != 0):
        print(L2)
        print(z2)
        p = L2[z2[0]-1] - L2[z2[-1]+1]
        add = p/(len(z2)+1)
        for i in z2:
            L2[i]=L2[i-1]+add
        k.gdp=L2
    #population
    if ((len(z3) >1) and evenly_spaced(z3) == True) and (z3[-1] != 18 and z3[0] != 0):
        p = L3[z3[0]-1] - L3[z3[-1]+1]
        add = p/(len(z3)+1)
        for i in z3:
            L3[i]=L3[i-1]+add
        k.population=L3
    return

#Applying the function for missing values
for v in filtered_data2['METROREG'].unique():
    treat_missing(filtered_data2[filtered_data2['METROREG'] == v])


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

# Print(filtered_data3)
