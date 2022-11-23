

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import datetime
import datetime as dt


##### EXO 2021 #####################################################################

#HYDRANT 3035

df_3035 = pd.read_csv('3035_FINAL.csv', squeeze=True) 

df_3035.loc[0:1538,'Date Time'] = pd.to_datetime(df_3035.loc[0:1538,'Date Time'],format='%d/%m/%y %H:%M:%S', utc=True)

df_3035.loc[1539:4994,'Date Time'] = pd.to_datetime(df_3035.loc[1539:4994,'Date Time'],format='%d/%m/%y %H:%M', utc=True)

df_3035.loc[4995:10178,'Date Time'] = pd.to_datetime(df_3035.loc[4995:10178,'Date Time'],format='%d/%m/%y %H:%M:%S', utc=True)

df_3035.loc[10179:,'Date Time'] = pd.to_datetime(df_3035.loc[10179:,'Date Time'],format='%d/%m/%y %H:%M', utc=True)

df_3035 = df_3035[1320:10920]

df_3035 = df_3035.set_index('Date Time')

new_cols = [ "Temperature"]

df_3035 = df_3035[new_cols]
df_3035=df_3035.reindex(columns=new_cols)

df_1 = df_3035[1320:10920]



df_3019 = pd.read_csv('3019_FINAL.csv', squeeze=True) 
df_3019.loc[0:2403,'Date Time'] = pd.to_datetime(df_3019.loc[0:2403,'Date Time'],format='%d/%m/%y %H:%M', utc=True)
df_3019.loc[2404:,'Date Time'] = pd.to_datetime(df_3019.loc[2404:,'Date Time'],format='%d/%m/%y %H:%M:%S', utc=True)
df_3019.drop(df_3019.index[6625:],inplace=True)


df_3019 = df_3019.set_index('Date Time')
new_cols = ["Temperature"]
df_3019 = df_3019[new_cols]

df_3019=df_3019.reindex(columns=new_cols)

df_2 = df_3019[209:]

print(df_2.isna().sum())



#date_range = pd.date_range(start="2021-10-04 15:40:00+00:00", end="2021-10-27 15:40:00+00:00", freq='5min')
#df_3019 = df_3019.reindex(date_range, fill_value=np.nan)

#HYDRANT 3044

df_3044 = pd.read_csv('3044_FINAL.csv', squeeze=True) 
df_3044.loc[0:2403,'Date Time'] = pd.to_datetime(df_3044.loc[0:2403,'Date Time'],format='%d/%m/%y %H:%M', utc=True)
df_3044.loc[2404:2691,'Date Time'] = pd.to_datetime(df_3044.loc[2404:2691,'Date Time'],format='%d/%m/%y %H:%M', utc=True)
df_3044.loc[2692:,'Date Time'] = pd.to_datetime(df_3044.loc[2692:,'Date Time'],format='%d/%m/%y %H:%M:%S', utc=True)
df_3044.drop(df_3044.index[6913:],inplace=True)
df_3044 = df_3044.set_index('Date Time')
#df_3035 = df_3035.resample('5T').ffill()

new_cols = [ "Temperature"]
df_3044 = df_3044[new_cols]
df_3044=df_3044.reindex(columns=new_cols)

df_3044 = df_3044[221:6912]

df_3 = df_3044[221:6912]


