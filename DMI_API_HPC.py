import numpy as np
import pandas as pd
import requests

data = pd.read_parquet(path = "Data/DSB_BDK_trainingset.parquet")
data = data.reset_index()

stations_data = pd.read_csv('Data/Stationskoder.csv', sep = ';')
stations_data = stations_data.rename(columns={'Nummer': 'station'})

merged = pd.merge(data[['dato', 'station']], stations_data[['station', '10km_cell']], on='station')

# Settings 10kmGridValue
api_key = 'cfaf5acf-a58b-44a2-af7c-793ca531edf1'
# metObsAPI: 'a5dfc496-b64b-4a35-9b04-685462e6e426'
# climateDataAPI: 'cfaf5acf-a58b-44a2-af7c-793ca531edf1'
DMI_URL = 'https://dmigw.govcloud.dk/v2/climateData/collections/10kmGridValue/items'
parameterIds = parameterIds = ['mean_temp', 'mean_wind_speed', 'acc_precip'] # Based on check
time_resolutions = ['day']
print(1)

dfs = []
for index, row in merged.iterrows():
    datetime_str = (row['dato'] + pd.DateOffset(days=-1)).tz_localize('UTC').isoformat() + '/' + (row['dato']).tz_localize('UTC').isoformat()
    cellId = row['10km_cell']
    station = row['station']
    dfii = {'station': station}
    for parameter in parameterIds:
        # Specify query parameters
        params = {
            'api-key' : api_key,
            'datetime' : datetime_str,
            'cellId' : cellId,
            'parameterId' : parameter,
            'limit' : '300000'  # max limit
            , 'timeResolution' : time_resolutions
        }

        # Submit GET request with url and parameters
        r = requests.get(DMI_URL, params=params)
        # Extract JSON object
        json = r.json() # Extract JSON object
        # Convert JSON object to a MultiIndex DataFrame and add to list
        dfi = pd.json_normalize(json['features'])
        if dfi.empty is False:
            # Drop other columns
            #dfi = pd.DataFrame({'station': station, 'parameterId': dfi['properties.parameterId'].values[0], 'value': dfi['properties.value'].values[0]}, index=[0])
            #dfi = pd.DataFrame({'station': station, dfi['properties.parameterId'].values[0] : dfi['properties.value'].values[0]}, index=[0])
            dfii[dfi['properties.parameterId'].values[0]] = dfi['properties.value'].values[0]
    dfs.append(pd.DataFrame(dfii, index=[0]))
    print(1)

df = pd.concat(dfs, axis='rows')


print(df)

df.to_csv('Data/DMI_data_precip.csv')