

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import datetime
import datetime as dt
from datetime import datetime, timedelta


##### EXO - April May #####################################################################

#GENERAL PRE-PROCESSING/FORMATING/PRELIMINARY OUTLIER REMOVAL:

df_exo_2022= pd.read_csv('EXO_GroupProject_2022.csv',squeeze=True) 

df_exo_2022.loc[0:3313,'Date Time'] = pd.to_datetime(df_exo_2022.loc[0:3313,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[3314:8497,'Date Time'] = pd.to_datetime(df_exo_2022.loc[3314:8497,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[8498:11953,'Date Time'] = pd.to_datetime(df_exo_2022.loc[8498:11953,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[11954:17425,'Date Time'] = pd.to_datetime(df_exo_2022.loc[11954:17425,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[11954:17425,'Date Time'] = pd.to_datetime(df_exo_2022.loc[11954:17425,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[17426:20881,'Date Time'] = pd.to_datetime(df_exo_2022.loc[17426:20881,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[20882:26065,'Date Time'] = pd.to_datetime(df_exo_2022.loc[20882:26065,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[26066:29521,'Date Time'] = pd.to_datetime(df_exo_2022.loc[26066:29521,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[29522:34993,'Date Time'] = pd.to_datetime(df_exo_2022.loc[29522:34993,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[34994:38449,'Date Time'] = pd.to_datetime(df_exo_2022.loc[34994:38449,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[38450:43921,'Date Time'] = pd.to_datetime(df_exo_2022.loc[38450:43921,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[43922:47376,'Date Time'] = pd.to_datetime(df_exo_2022.loc[43922:47376,'Date Time'], utc=True,dayfirst=True)
df_exo_2022.loc[47377:47377,'Date Time'] = pd.to_datetime(df_exo_2022.loc[47377:47377,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[47378:52561,'Date Time'] = pd.to_datetime(df_exo_2022.loc[47378:52561,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[52562:56017,'Date Time'] = pd.to_datetime(df_exo_2022.loc[52562:56017,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[56018:61489,'Date Time'] = pd.to_datetime(df_exo_2022.loc[56018:61489,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[61490:64945,'Date Time'] = pd.to_datetime(df_exo_2022.loc[61490:64945,'Date Time'], utc=True,dayfirst=True)

df_exo_2022.loc[64946:,'Date Time'] = pd.to_datetime(df_exo_2022.loc[64946:,'Date Time'], utc=True,dayfirst=True)


df_exo_2022.loc[43922:,'Date Time'] = pd.to_datetime(df_exo_2022.loc[43922:,'Date Time'],format='%d/%m/%y %H:%M:%S', utc=True,dayfirst=True)


#DEFINE DATA FRAMES 
df_4 = df_exo_2022[2:15020]
df_5 = df_exo_2022[17300:23930]
df_6 = df_exo_2022[25095:]

df_exo_2022 = df_exo_2022[44786:].set_index('Date Time')
df_exo_2022.plot()

max_temp = max(df_exo_2022 ['Temperature'])

min_temp = min(df_exo_2022 ['Temperature'])








