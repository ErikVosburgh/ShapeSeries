# -*- coding: utf-8 -*-


#%%## Initialization

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import calinski_harabasz_score 
from EXO_2022 import df_exo_2022
from EXO_2021 import df_1, df_2, df_3
from helper_functions import shape_scan, kmeans_ts, kmedoids_ts, load_ts_data

from store_prototype_temp import shape_prototype

param = 'Temperature'
units = 'Deg. C'

color_list = ['purple', 'red', 'green', 'violet', 'pink', 'blue', 'black', 'orange', 'teal', 'brown', 'olive', 'cyan','yellow']
df_online_1, df_online_2, df_online_3, df_offline, pattern_df = load_ts_data()
data_scaled = df_offline
data_scaled = data_scaled.reset_index()
#data_scaled = data_scaled.drop(columns=['index'])

#%% DEFINE CLUSTER PARAMETERS 
slide_len_hrs = 24
slide_len = slide_len_hrs*12
hrs = 24
m = hrs*12

sensor = data_scaled.loc[:,'scaled']
sensor = sensor.interpolate(method='pad')
#sensor.plot()
sensor = sensor.values #convert to array
  
  
  #%%## EXTRACT ALL SHAPES AND STORE IN DATA FRAME
ss_df = []
for start_pos in range(0, len(sensor), slide_len):
    end_pos = start_pos + m
    segment = np.copy(sensor[start_pos:end_pos])
    if len(segment) != m:
        continue
    ss_df.append(segment)

ss_df = pd.DataFrame(ss_df)

X = np.c_[ss_df]
  
  #%% Step 2: set up and create k-means model
  
                           # define the number of clusters
percentile = 5

####################################################################################
training_data = data_scaled

pattern = 0
idx_start = 1350
Temperature_shape_1,Temperature_shape_1_thresh, Temperature_pattern_1, matches_pattern_1,dist_pattern_1 = shape_prototype(training_data,idx_start,param,units,color_list[pattern], percentile)
shape_1_median = dist_pattern_1.median()
shape_1_std = round(dist_pattern_1.std(),3)
shape_1_thresh = round(dist_pattern_1.max(),3)


pattern = 1
idx_start = 8000
Temperature_shape_2,Temperature_shape_2_thresh, Temperature_pattern_2, matches_pattern_2,dist_pattern_2 = shape_prototype(training_data,idx_start,param,units,color_list[pattern], percentile)
shape_2_median = dist_pattern_2.median()
shape_2_std = round(dist_pattern_2.std(),3)
shape_2_thresh = round(dist_pattern_2.max(),3)

pattern = 2
idx_start = 10000
Temperature_shape_3,Temperature_shape_3_thresh, Temperature_pattern_3, matches_pattern_3,dist_pattern_3 = shape_prototype(training_data,idx_start,param,units,color_list[pattern], percentile)
shape_3_median = dist_pattern_3.median()
shape_3_std = round(dist_pattern_3.std(),3)
shape_3_thresh = round(dist_pattern_3.max(),3)
####################################################################################
####################################################################################

pattern = 3
idx_start = 13000
Temperature_shape_4,Temperature_shape_4_thresh, Temperature_pattern_4, matches_pattern_4,dist_pattern_4 = shape_prototype(training_data,idx_start,param,units,color_list[pattern], percentile)
shape_4_median = dist_pattern_4.median()
shape_4_std = round(dist_pattern_4.std(),3)
shape_4_thresh = round(dist_pattern_4.max(),3)
####################################################################################

pattern = 4
idx_start = 16200
Temperature_shape_5,Temperature_shape_5_thresh, Temperature_pattern_5, matches_pattern_5,dist_pattern_5 = shape_prototype(training_data,idx_start,param,units,color_list[pattern], percentile)
shape_5_median = dist_pattern_5.median()
shape_5_std = round(dist_pattern_5.std(),3)
shape_5_thresh = round(dist_pattern_5.max(),3)
####################################################################################

  
  #%%# Combine prototypes and summary stats into dataframes
  
centroids_manual = pd.concat([Temperature_shape_1,Temperature_shape_2,Temperature_shape_3,Temperature_shape_4,Temperature_shape_5], axis=1)
columns = ['0','1','2','3','4']
centroids_manual.columns = columns

cluster_matches = pd.concat([matches_pattern_1,matches_pattern_2,matches_pattern_3,matches_pattern_4,matches_pattern_5,data_scaled['Date Time']], axis=1)
columns = ['0','1','2','3','4','Date_Time']
cluster_matches.columns = columns
cluster_matches = cluster_matches.set_index(['Date_Time'])

cluster_distance= pd.concat([dist_pattern_1,dist_pattern_2,dist_pattern_3,dist_pattern_4,dist_pattern_5, data_scaled['Date Time']], axis=1)
columns = ['0','1','2','3','4','Date_Time']
cluster_distance.columns = columns
cluster_distance = cluster_distance.set_index(['Date_Time'])

thresh_manual= pd.Series([shape_1_thresh,shape_2_thresh,shape_3_thresh,shape_4_thresh,shape_5_thresh]).to_frame()
thresh_manual = thresh_manual.transpose()

std_manual= pd.Series([shape_1_std,shape_2_std,shape_3_std,shape_4_std,shape_5_std]).to_frame()
std_manual = std_manual.transpose()

  
  #%%# Histogram of matches

fig, axes = plt.subplots(nrows = 5, ncols =1, figsize = (8, 10))
for index, column in enumerate(cluster_distance.columns):
    ax = axes.flatten()[index]
    ax.hist(cluster_distance[column], color = color_list[index], label = 'pattern {}, thresh = {}, stdev. = {}'.format(column,thresh_manual.loc[0,index],std_manual.loc[0,index]))
    ax.legend(loc = "best")
plt.suptitle("Matches below threshold for manually selected patterns", size = 20)
plt.show()
