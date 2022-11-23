# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 14:18:07 2020

@author: Lina Sela

lectures 16 - 18: k-means
"""

#%%## Initialization

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from EXO_2022 import *
from EXO_2021 import df_1, df_2, df_3
from sklearn.cluster import KMeans
from param_norm import*

color_list = ['purple', 'red', 'green', 'violet', 'pink', 'blue', 'black', 'orange', 'teal', 'brown', 'olive', 'cyan','yellow']

#%%## IMPORT AND FORMAT DATA 
param = 'Temperature'
unit = 'Deg. C'

df_online_1 = param_norm(param,df_1)['scaled']
df_online_2 = param_norm(param,df_2)['scaled']
df_online_3 = param_norm(param,df_3)['scaled']
df_offline = param_norm(param,df_exo_2022)['scaled']


color_list = ['purple', 'red', 'green', 'yellow', 'pink', 'blue', 'black', 'violet', 'teal', 'brown', 'olive', 'cyan','orange']

df_offline = df_offline.reset_index()
sensor =df_offline.loc[:,'scaled']
sensor = sensor.interpolate(method='pad')
sensor.plot()
sensor = sensor.values #convert to array


#%%## SPLIT TS INTO PATTERNS, STORE IN DATA FRAME
pattern_df = []
slide_len = 288
hrs = 24
m = hrs*12
for start_pos in range(0, len(sensor), slide_len):
    end_pos = start_pos + m
    # make a copy so changes to 'segments' doesn't modify the original ekg_data
    segment = np.copy(sensor[start_pos:end_pos])
    # if we're at the end and we've got a truncated segment, drop it
    if len(segment) != m:
        continue
    pattern_df.append(segment)
pattern_df = pd.DataFrame(pattern_df)

df_offline = df_offline.set_index(['Date Time'])




