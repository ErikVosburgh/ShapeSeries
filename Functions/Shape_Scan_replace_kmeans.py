import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Group_4_sensor_plots import *
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids 
import stumpy
from Shape_Scan_median_k import *
from kmeans_sensors import *


#%%
param = 'Temperature'
units = 'Deg. C'
begin = '2022-09-04 00:00:00+00:00'
end = 	'2022-11-15 14:10:00+00:00'

color_list = ['purple', 'red', 'green', 'violet', 'pink', 'blue', 'black', 'orange', 'teal', 'brown', 'olive', 'cyan','yellow']
data_scaled = df_offline

#%% Get Clusters ##
slide_len_hrs = 24
slide_len = slide_len_hrs*12
hrs = 24
m = hrs*12
k = 6 
thresh_df,std_df,centroids = kmeans_ts(data_scaled,k,slide_len,m, color_list)

#%% run matrix profile ##

mp = stumpy.stump(data_scaled['scaled'], m, normalize=False)
mp = pd.DataFrame(mp)
a = mp.loc[:,0]
data_t = data_scaled.reset_index()
b = data_t.loc[:,'Date Time']
mp_df = pd.concat([a,b], axis=1)

#data_scaled = data_scaled.set_index(['Date Time'])
mp_df = mp_df.set_index(['Date Time'])


#%% SCAN
target_data = data_scaled 
match_idx = {}
shape = {}
shape_dist_fltrd = {}
shape_dist = {}

for i in range(0,k): 
    match_idx["{0}".format(i)],shape["{0}".format(i)],ts,shape_dist_fltrd["{0}".format(i)],shape_dist["{0}".format(i)] = shape_scan(centroids[i],target_data,param,units,color_list[i],thresh_df.iloc[0,i])
    
shape_df = pd.DataFrame.from_dict(shape)
shape_dist_fltrd_df = pd.DataFrame.from_dict(shape_dist_fltrd)
shape_dist_df = pd.DataFrame.from_dict(shape_dist)

mask = {}
median  = {}

for i in range(0,k):
    mask["{0}".format(i)] = shape_df.iloc[:,i].isnull()
    median["{0}".format(i)] = shape_df.iloc[:,i].median()
    
mask_df = pd.DataFrame.from_dict(mask)

#%% 
##### CREATE NEW DATAFRAME OF SHAPE MATCHES #################################################################################################################
data=target_data                       
data = data.reset_index()
test = data.loc[:,'Date Time']

##### Resolve minimum value conflicts #################################################################################################################
symbol_check = shape_dist_fltrd_df.copy()
df_temp = shape_dist_fltrd_df.copy()
symbol_series= df_temp.idxmin(axis=1)
symbol_series_post = symbol_series.copy()
not_nan = symbol_series.index[~symbol_series.isnull()]
for item in not_nan:
    symbol_series_post[item:item+m] = symbol_series[item]
    
dist_bool = symbol_series
dist_bool.fillna(True,inplace=True)


#%% 
#####################################################################################

ts_df = pd.concat([data.loc[:,'Date Time'],ts,shape_df], axis=1)
symbol_df = ts_df.copy()

for i in range(0,k):
    symbol_df['{0}'.format(i)].loc[symbol_series_post == '{0}'.format(i)] = median.get('{}'.format(i))
    symbol_df['{0}'.format(i)].loc[symbol_series_post != '{0}'.format(i)] = np.nan
    symbol_df[param] = symbol_df[param].mask(~mask_df['{0}'.format(i)])    

#%% FUZZY
#####################################################################################
symbol_df_fuzzy = symbol_df.copy()
symbol_df_fuzzy = symbol_df_fuzzy.set_index('Date Time')
symbol_df_fuzzy = symbol_df_fuzzy.drop([param],axis=1)
symbol_df_fuzzy.loc[:] = np.nan
symbol_df_fuzzy = symbol_df_fuzzy.reset_index()

#rows = symbol_df_fuzzy.index[symbol_df_fuzzy.loc[:,'shape_1':'shape_6'].isnull().all(1)]


#pattern = 0

for i in range(0,k):
    symbol_df_fuzzy['{0}'.format(i)].iloc[shape_dist_df['{0}'.format(i)] <=(thresh_df.iloc[0,i]+ (2*std_df.iloc[0,i])) ] = median.get('{}'.format(i))
      

mp_df_mask = mp_df.copy()
mp_df_mask = mp_df_mask.reset_index()
mp_df = mp_df.reset_index()

mp_bool = symbol_series_post.copy()
mp_bool.fillna(True,inplace=True)

mp_df_mask[0].iloc[mp_bool != True] = np.nan

symbol_df_fuzzy = symbol_df_fuzzy.set_index('Date Time')
symbol_df_fuzzy.iloc[symbol_series_post.notna()] = np.nan

#%% FILTER COMPETING FUZZY MATCHES 
#####################################################################################
mp95 = np.percentile(mp_df[0],50)

min_df = np.nanmin(symbol_df_fuzzy,axis=1)
min_df = pd.DataFrame(min_df )
symbol_df_fuzzy = symbol_df_fuzzy.reset_index()

for i in range(0,k):
    symbol_df_fuzzy['{0}'.format(i)].iloc[symbol_df_fuzzy['{0}'.format(i)]>(min_df[0])] = np.nan

symbol_df_fuzzy = symbol_df_fuzzy.set_index('Date Time')
symbol_df_fuzzy_anom = symbol_df_fuzzy.copy()

tsfuzzy = symbol_df[param].copy().to_frame()
mpfuzzy = mp_df[0].copy().to_frame()

tsfuzzy.loc[(mpfuzzy[0]<=mp95)]=np.nan
tsfuzzy=pd.concat([tsfuzzy,b],axis=1)
tsfuzzy.set_index(['Date Time'],inplace=True)

ts_df= ts_df.reset_index()

ts_df['shape_1'].loc[symbol_series_post != 'shape_1'] = np.nan

for index in enumerate(match_idx_1):
    index = index[1]  
    if symbol_series_post[index]=='shape_1':
        ts_df['shape_1'].loc[index] = ts_df['ts'].loc[index]
    
ts_df['shape_2'].loc[symbol_series_post != 'shape_2'] = np.nan
ts_df['shape_2'].loc[symbol_series_post == 'shape_2'] = ts_df['ts']
# =============================================================================

ts_df['shape_3'].loc[symbol_series_post != 'shape_3'] = np.nan
ts_df['shape_3'].loc[symbol_series_post == 'shape_3'] = ts_df['ts']
# =============================================================================

ts_df['shape_4'].loc[symbol_series_post != 'shape_4'] = np.nan
ts_df['shape_4'].loc[symbol_series_post == 'shape_4'] = ts_df['ts']
# =============================================================================

ts_df['shape_5'].loc[symbol_series_post != 'shape_5'] = np.nan
ts_df['shape_5'].loc[symbol_series_post == 'shape_5'] = ts_df['ts']
# =============================================================================

ts_df['shape_6'].loc[symbol_series_post != 'shape_6'] = np.nan
ts_df['shape_6'].loc[symbol_series_post == 'shape_6'] = ts_df['ts']
# =============================================================================


ts_df= ts_df.set_index(['Date Time'])
ts_df= ts_df.drop(['index'], axis=1)
symbol_df= symbol_df.set_index(['Date Time'])
mp_df_mask = mp_df_mask.set_index(['Date Time'])
mp_df = mp_df.set_index(['Date Time'])
# =============================================================================

nan_count = symbol_series_post.count()
length = len(symbol_series_post)

pcnt_match = round(((nan_count)/len(symbol_series_post))*100,2)

#######################################################################

fig, axes = plt.subplots(nrows=4, ncols=1)

ts_df.plot(y=[param], ax=axes[0], color = 'grey', alpha=1, use_index=True, subplots=True, legend=False)
tsfuzzy[param].plot(y=[param], ax=axes[0], color = 'black', alpha=1, use_index=True, subplots=True)
ts_df.plot(y=['0'], ax=axes[0], color = color_list[0], alpha=1, use_index=True, subplots=True, legend=False)
ts_df.plot(y=['1'], ax=axes[0], color = color_list[1], alpha=1, use_index=True, subplots=True, legend=False)
ts_df.plot(y=['2'], ax=axes[0], color = color_list[2], alpha=1, use_index=True, subplots=True, legend=False)
ts_df.plot(y=['3'], ax=axes[0], color = color_list[3], alpha=1, use_index=True, subplots=True, legend=False)
ts_df.plot(y=['4'], ax=axes[0], color = color_list[4], alpha=1, use_index=True, subplots=True, legend=False)
ts_df.plot(y=['5'], ax=axes[0], color = color_list[5], alpha=1, use_index=True, subplots=True, legend=False)

# =============================================================================

symbol_df[param].plot(y=[param], ax=axes[1], color = 'grey', alpha=1, use_index=True, subplots=True)
tsfuzzy[param].plot(y=[param], ax=axes[1], color = 'black', alpha=1, use_index=True, subplots=True)
symbol_df['0'].plot(y=['0'], ax=axes[1], color = color_list[0],linestyle='none', alpha=1, use_index=True, subplots=True, markersize=2,markevery=5)
symbol_df.plot(y=['1'], ax=axes[1], color = color_list[1],linestyle='none', alpha=1,marker='x', use_index=True, subplots=True, markersize=2,markevery=5)
symbol_df.plot(y=['2'], ax=axes[1], color = color_list[2],linestyle='none', alpha=1,marker='x', use_index=True, subplots=True, markersize=2,markevery=5)
symbol_df.plot(y=['3'], ax=axes[1], color = color_list[3],linestyle='none', alpha=1,marker='x', use_index=True, subplots=True, markersize=2,markevery=5)
symbol_df.plot(y=['4'], ax=axes[1], color = color_list[4],linestyle='none', alpha=1,marker='x', use_index=True, subplots=True, markersize=2,markevery=5)
symbol_df.plot(y=['5'], ax=axes[1], color = color_list[5],linestyle='none', alpha=1,marker='x', use_index=True, subplots=True,markersize=2,markevery=5)

tsfuzzy[param].plot(y=[param], ax=axes[2], color = 'black', alpha=1, use_index=True, subplots=True)
symbol_df_fuzzy['0'].plot(y=['0'], ax=axes[2], color = color_list[0],linestyle='none', alpha=1, subplots=True,marker = 'X', markersize=2,markevery=20)
symbol_df_fuzzy.plot(y=['1'], ax=axes[2], color = color_list[1],linestyle='none', alpha=1,  subplots=True,marker = 'X', markersize=4,markevery=20)
symbol_df_fuzzy.plot(y=['2'], ax=axes[2], color = color_list[2],linestyle='none', alpha=1,  subplots=True,marker = 'X', markersize=4,markevery=20)
symbol_df_fuzzy.plot(y=['3'], ax=axes[2], color = color_list[3],linestyle='none', alpha=1,  subplots=True,marker = 'X', markersize=4,markevery=20)
symbol_df_fuzzy.plot(y=['4'], ax=axes[2], color = color_list[4],linestyle='none', alpha=1,  subplots=True,marker = 'X', markersize=4,markevery=20)
symbol_df_fuzzy.plot(y=['5'], ax=axes[2], color = color_list[5],linestyle='none', alpha=1, subplots=True,marker = 'X', markersize=4,markevery=20)

mp_df.plot(ax=axes[3], color = 'grey', alpha=0.5, use_index=True, subplots=True)
mp_df_mask.plot(ax=axes[3], color = 'grey', alpha=1, use_index=True, subplots=True)



axes[0].legend(bbox_to_anchor=(1, 0.5), prop={"size":10})
axes[1].get_legend().remove()
axes[2].get_legend().remove()
axes[3].get_legend().remove()
axes[0].get_legend().remove()

axes[0].set_xticks([])
axes[1].set_xticks([])
axes[2].set_xticks([])

#axes[3].hlines(y=mp95, linewidth=2, color='black')


axes[0].set_xlim(begin, end)
axes[1].set_xlim(begin, end)
axes[2].set_xlim(begin, end)
axes[3].set_xlim(begin, end)
axes[2].set_ylim(bottom=0,top=1)
#axes[3].set_xticklabels(axes[3].get_xticks(), rotation = 45)
#axes[0].set_ylabel("{} - |n standardized".format(units))
axes[1].set_ylabel("{} - standardized".format(units))
#axes[2].set_ylabel("{}- |n standardized".format(units))
axes[3].set_ylabel("Distance".format(units))
fig.suptitle('{}% of subsequences \n are a match'.format(pcnt_match))


plt.savefig('Shape_Scan_Replace.png',dpi=300, bbox_inches = "tight") 

plt.show()
###############33##############3##############################################
fig, axes = plt.subplots(nrows=2)

ts_df['ts'].plot(y=['ts'], ax=axes[0], color = 'grey', alpha=1, use_index=True, subplots=True)

symbol_df['ts'].plot(y=['ts'], ax=axes[1], color = 'grey', alpha=1, use_index=True, subplots=True)
symbol_df['shape_1'].plot(y=['shape_1'], ax=axes[1], color = 'purple',linestyle='none', alpha=1, use_index=True, subplots=True,marker = symbol_1, markersize=2,markevery=5)
symbol_df.plot(y=['shape_2'], ax=axes[1], color = 'red',linestyle='none', alpha=1, use_index=True, subplots=True,marker = symbol_2, markersize=2,markevery=5)
symbol_df.plot(y=['shape_3'], ax=axes[1], color = 'green',linestyle='none', alpha=1, use_index=True, subplots=True,marker = symbol_3, markersize=2,markevery=5)
symbol_df.plot(y=['shape_4'], ax=axes[1], color = 'yellow',linestyle='none', alpha=1, use_index=True, subplots=True,marker = symbol_4, markersize=2,markevery=5)
symbol_df.plot(y=['shape_5'], ax=axes[1], color = 'pink',linestyle='none', alpha=1, use_index=True, subplots=True,marker = symbol_2, markersize=2,markevery=5)
symbol_df.plot(y=['shape_6'], ax=axes[1], color = 'blue',linestyle='none', alpha=1, use_index=True, subplots=True,marker = symbol_6, markersize=2,markevery=5)
axes[1].legend(bbox_to_anchor=(1, 1.2), prop={"size":10})
axes[1].set_ylabel("{} - standardized".format(units))
axes[0].set_ylabel("{} - standardized".format(units))
plt.savefig('Shape_Scan_Replace_abrv.png',dpi=300, bbox_inches = "tight") 
plt.show()
#######################################################################

idx = 37779
data=data.reset_index()
fig, axes = plt.subplots(nrows=3, ncols=2)
target = (data['scaled'][idx:idx+m].to_frame()).reset_index()
target = target.drop(columns = ['index'])

p1_dist = round(dist_df.iloc[idx,0],2)
p2_dist = round(dist_df.iloc[idx,1],2)
p3_dist = round(dist_df.iloc[idx,2],2)
p4_dist = round(dist_df.iloc[idx,3],2)
p5_dist = round(dist_df.iloc[idx,4],2)
p6_dist = round(dist_df.iloc[idx,5],2)

target_prototype = pd.concat([centroids,target], axis=1)

dist_df.iloc[0,1]

#ORP_medianshape_library.plot(y=['1'], ax=axes[0,0], color = 'purple', linestyle='dashed', alpha=1, label='Pattern thresh is {} \n, distance is {}'.format(ORP_shape_1_thresh,p1_dist))
target_prototype.plot(y=[0], ax=axes[0,0], color = 'purple', linestyle='dashed', alpha=0.7)
target_prototype.plot(y=[1], ax=axes[0,1], color = 'red', linestyle='dashed', alpha=0.7)
target_prototype.plot(y=[2], ax=axes[1,0], color = 'green', linestyle='dashed', alpha=0.7)
target_prototype.plot(y=[3], ax=axes[1,1], color = 'yellow', linestyle='dashed', alpha=0.7)
target_prototype.plot(y=[4], ax=axes[2,0], color = 'pink', linestyle='dashed', alpha=0.7)
target_prototype.plot(y=[5], ax=axes[2,1], color = 'blue', linestyle='dashed', alpha=0.7)

target_prototype.plot(y=['scaled'], ax=axes[0,0], color = 'grey', alpha=1)
target_prototype.plot(y=['scaled'], ax=axes[0,1], color = 'grey', alpha=1)
target_prototype.plot(y=['scaled'], ax=axes[1,0], color = 'grey', alpha=1)
target_prototype.plot(y=['scaled'], ax=axes[1,1], color = 'grey', alpha=1)
target_prototype.plot(y=['scaled'], ax=axes[2,0], color = 'grey', alpha=1)
target_prototype.plot(y=['scaled'], ax=axes[2,1], color = 'grey', alpha=1)



axes[0,0].legend( title='Shape 1 thresh. = {} \n distance is {}'.format(round(thresh_df.iloc[0,0],2),p1_dist), fontsize=0.5,bbox_to_anchor=(-.2, 1))
axes[0,1].legend( title='Shape 2 thresh. = {} \n distance is {}'.format(round(thresh_df.iloc[0,1],2),p2_dist), fontsize=0.5,bbox_to_anchor=(1.2, 1))
axes[1,0].legend( title='Shape 3 thresh. = {} \n distance is {}'.format(round(thresh_df.iloc[0,2],2),p3_dist), fontsize=0.5,bbox_to_anchor=(-.2, 1))
axes[1,1].legend( title='Shape 4 thresh. = {} \n distance is {}'.format(round(thresh_df.iloc[0,3],2),p4_dist), fontsize=0.5,bbox_to_anchor=(1.2, 1))
axes[2,0].legend( title='Shape 5 thresh. = {} \n distance is {}'.format(round(thresh_df.iloc[0,4],2),p5_dist), fontsize=0.5,bbox_to_anchor=(-.2, 1))
axes[2,1].legend( title='Shape 6 thresh. = {} \n distance is {}'.format(round(thresh_df.iloc[0,5],2),p6_dist), fontsize=0.5,bbox_to_anchor=(1.2, 1))


plt.savefig('Target_prototype_compare.png',dpi=300, bbox_inches = "tight") 
plt.show()

#############################################################################
start = idx - 2000
# end = idx + 2000
ts = ts_df.reset_index()
ts = ts[['ts','Date_Time']].copy()
target_ts =np.empty(len(ts))
target_ts[:] = np.nan
match = symbol_series_post.loc[idx]

target_ts[idx:idx+m] = ts.iloc[idx:idx+m,0]


target_ts = pd.DataFrame(target_ts)
target_ts = pd.concat([target_ts,ts], axis=1)
target_ts = target_ts.iloc[start:end]
target_ts = target_ts.set_index('Date_Time')

#fig, axes = plt.subplots()
fig, axes = plt.subplots()

target_ts.plot(y=['ts'], color = 'grey', alpha=0.5, use_index=True, subplots=True, ax=axes)
target_ts.plot(y = [0],color = 'black',ax=axes)

axes.legend( title='{} is the best match'.format(match), fontsize=2)

plt.savefig('Target_prototype_plot.png',dpi=300, bbox_inches = "tight") 

plt.show


