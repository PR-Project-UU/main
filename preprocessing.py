import pandas as pd
import glob
import numpy as np
import geopy
from geopy.geocoders import Nominatim
from functools import partial
import scipy


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
filtered_data2.population = filtered_data2.population.replace(':', np.nan) # replacing NAs (noted ':') by NaN
filtered_data2.population = filtered_data2['population'].replace(r',', '',regex=True)  # removing commas
filtered_data2.population = filtered_data2['population'].astype(float)  #changing column type

# Preprocessing on the GDP column
filtered_data2.gdp = filtered_data2.gdp.replace(':',np.nan )  # replacing missing values (noted ':') by NaN
filtered_data2.gdp = filtered_data2['gdp'].replace(r',', '',regex=True)  # removing commas
filtered_data2.gdp = filtered_data2['gdp'].astype(float)  # changing column type

# Employed_persons column
filtered_data2.employed_persons = filtered_data2.employed_persons.replace(':', np.nan)
filtered_data2.employed_persons = filtered_data2['employed_persons'].replace(r',', '',regex=True)
filtered_data2.employed_persons = filtered_data2['employed_persons'].astype(float)

#removes cities for which we have too much missing values-> no point in doing a linear rgression
cities_removed = []
for v in filtered_data2['METROREG'].unique():
    if max(list(filtered_data2[filtered_data2['METROREG'] == v].isnull().sum()))>8:
        cities_removed.append(v)
for i in cities_removed: 
    indexMetroreg = filtered_data2[filtered_data2['METROREG'] == i].index
    filtered_data2.drop(indexMetroreg, inplace=True)

#PERFORMS LINEAR REGRESSION TO REPLACE MISSING VALUES

from scipy.optimize import curve_fit

def F(x,a,b):
    return a*x+b

def nan_position(k):
    z=np.argwhere(np.isnan(np.array(k)))
    l=[]
    for i in z:
        l.append(i[0])
    return l

def treat_missing(k):
    #employed_persons
    missing=k.isnull().sum()
    if missing[2] != 0: #first linear regression
        L1=list(k.employed_persons)
        m1=nan_position(k.employed_persons) #gets position of missing
        X1=[i for i in range(len(list(k.employed_persons))) if str(list(k.employed_persons)[i])!='nan']
        Y1=[i for i in list(k.employed_persons) if str(i)!='nan']
        params1 = curve_fit(F, xdata=X1, ydata=Y1) #performs linear regression, params[0] contains a and b
        for i in m1:
            L1[i] = params1[0][0]*i + params1[0][1]
        k.employed_persons = L1
        
    if missing[3] != 0: #second linear regression
        L2=list(k.gdp)
        m2=nan_position(k.gdp) #gets position of missing
        X2=[i for i in range(len(list(k.gdp))) if str(list(k.gdp)[i])!='nan']
        Y2=[i for i in list(k.gdp) if str(i)!='nan']
        params2 = curve_fit(F, xdata=X2, ydata=Y2) #performs linear regression, params[0] contains a and b
        for i in m2:
            L2[i] = params2[0][0]*i + params2[0][1]
        k.gdp=L2 #replacing the column
    
    if missing[4] != 0: #third linear regression
        L3=list(k.population)
        m3=nan_position(k.population) #gets position of missing
        X3=[i for i in range(len(list(k.population))) if str(list(k.population)[i])!='nan']
        Y3=[i for i in list(k.population) if str(i)!='nan']
        params3 = curve_fit(F, xdata=X3, ydata=Y3) #performs linear regression, params[0] contains a and b
        for i in m3:
            L3[i] = params3[0][0]*i + params3[0][1]
        k.population=L3#replacing the column
    return k

#applying function above

INDEX = filtered_data2['METROREG'].unique()
filtered_data3 = pd.DataFrame(columns=['TIME', 'METROREG', 'employed_persons', 'gdp', 'population'])
for i in range(len(INDEX)):
    i = filtered_data2[filtered_data2['METROREG'] == INDEX[i]].copy(deep = True)
    filtered_data3 = filtered_data3.append(treat_missing(i))

# OPTIMAL VALUE TO BE DETERMINED
# As of now, we look at cities with population>500000 ,
# could be changed if we need more/less data
filtered_data3 = filtered_data3[(filtered_data3['TIME'] >= 2000) &
                                (filtered_data3['population'] > 500000)]

###### Geolocation part

"""
To bridge the images and the socioeconomic data, we can use the geopy package to retrieve the latitude & 
longitude of the cities in our data set. From there, we need to figure out a way to access the satellite 
images w/latitude & longitude
"""


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
cities_removed2=[]
for city in sorted(list(set(list(filtered_data3.METROREG)))): #[:5]:  # [:5] is voor debug, weghalen later
    try:
        city_coords = get_coord(city)
        coords_list.append([city, city_coords])
    except:
        cities_removed2.append(city)
        print("Couldn't get coords of:", city)

#removing cities for which we couldn't get the coordinates
for i in cities_removed2:
    filtered_data3=filtered_data3[filtered_data3.METROREG!=i]

# Adding corresponding positions to cities
for coords_city in coords_list:
    city = coords_city[0]
    coords = coords_city [1]
    filtered_data3.loc[filtered_data3['METROREG'] == city, 'latitude'] = coords[0]
    filtered_data3.loc[filtered_data3['METROREG'] == city, 'longitude'] = coords[1]

filtered_data3.to_csv('./meta_features.csv')
