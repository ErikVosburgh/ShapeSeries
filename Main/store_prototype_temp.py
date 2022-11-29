
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from scipy.stats import iqr
import math


#training_data = data_scaled
#idx_start = 1350
#param = 'Temperature'
#units = 'units'
#percentile = 5
#color_1 = 'blue'

### LOAD TEMPLATE SHAPE DATA: ##########################################################################
def shape_prototype(data,idx_start,param,units,color_1, percentile):

    #data = data.reset_index()
    data_target = data
    #data = data.interpolate(method='pad')
    param = param
    unit = units
    
    #NORMALIZE 
    data = data_target
    #data_min = min(data[param])
    #data_max = max(data[param])
    #data['scaled'] = [(num - data_min) / (data_max - data_min) for num in data[param]]
    
    pattern_length = 24 #hrs
    m = pattern_length*12
    idx_start = idx_start
    shape_data = data['scaled'][idx_start:idx_start+m]
    
    #axes = shape_data.plot(y=['scaled'], color = [ 'blue'], alpha=0.5, use_index=True, subplots=True)
    #axes[1].tick_params(labelrotation=45)
    #plt.show()
           
    ###store shape in data frame 
    shape_data = data[idx_start:idx_start+m]
    shape_df = pd.DataFrame(columns=['shape'])
    shape_df['shape'] = shape_data['scaled']
    pattern_n_df = pd.DataFrame(index = data.index, columns=['pattern_n'])
    pattern_n_df.loc[idx_start:idx_start+m,'pattern_n'] = data.loc[idx_start:idx_start+m,'scaled']
    
    
    ### LOAD TEMPLATE/TARGET DATAFRAME: ##########################################################################
    data = data
    #data = data.reset_index()
    #data = data.interpolate(method='pad')
    #data['scaled'] = [(num - data_min) / (data_max - data_min) for num in data[param]]
    
    ###store ts and distance profile in data frame
    target_df = pd.DataFrame(columns=[param,'ed_profile','dist','match'])
    target_df[param] = data['scaled']
    
    ed_profile = np.ones(len(data))*np.inf
    for i in range(0, len(data)  - m - 1): #assign and standardize sub_stream from template sequence
        sub_stream = target_df.loc[i:i + m - 1,param]
        sub_stream=sub_stream.to_numpy()    
        ed_profile[i] = np.linalg.norm(sub_stream - shape_df['shape'].to_numpy()) 
        
    target_df['ed_profile'] = ed_profile
    target_df['dist'] = ed_profile
    
    ### IDENTIFY MATCHES ##########################################################################
    
    bottom = np.nanpercentile(target_df['ed_profile'],percentile)
    
    target_df['ed_profile'] = target_df['ed_profile'].mask(target_df['ed_profile'] > bottom)
    
    condition = target_df['ed_profile'] <= bottom
    
    ### New TS for TS 1 matches 
    for i in range(0, len(data)-m):
        if target_df.loc[i,'ed_profile']<=bottom: 
            target_df.loc[i:i+m,'shape'] = target_df.loc[i:i+m,param] 
                
    ###########################################################################################
    # Store and Plot shapes 
    
    x = np.arange(0,m,1)
    i=0
    j=0
    
    matches = sum(condition)
    matches = pd.DataFrame(index=np.arange(0,m,1), columns=np.arange(matches))
    match_idx = []
    
    
    length = sum(condition)
    
    while i < len(condition):

        if math.isnan(target_df.loc[i,'ed_profile'])==False: 

            matches.iloc[0:0+m,j] = data.loc[i:i+m-1,'scaled']
            match_idx.append(i)
            j += 1 
             
            i += m
        i += 1 
    
    matches = matches.dropna(axis=1, how='all')
    shape_df = shape_df.reset_index()
    shape_df = shape_df.drop(columns=['index'])
    shape_df = pd.concat([shape_df,matches], axis=1)
    
    cols = [col for col in shape_df.columns]
    
    shape_df['median'] = shape_df[cols].median(axis=1)
    
    target_df = pd.concat([target_df,pattern_n_df], axis=1)
    
    ## PLOT SHAPES #########################################################################################
    
    fig, ax = plt.subplots()
    shape_df.plot(figsize=(20,12), title='Matching Subsequences', 
            lw=3, fontsize=16, ax=ax, grid=True, color = 'grey')
    
    for line in ax.get_lines():
        if line.get_label() == 'median':
            line.set_linewidth(10)
            line.set(color=color_1)
            line.set(linestyle='--')
        if line.get_label() == 'shape':
            line.set_linewidth(10)
            line.set(color=color_1)
    plt.savefig('shape_median',dpi=300, bbox_inches = "tight") 
    plt.show()
    
    ###########################################################################################
    
    fig, axes = plt.subplots(nrows=2, ncols=1)
    
    # First subplot:
    
    target_df.plot(y=[ param], ax=axes[0], color = [ 'grey'], alpha=1, use_index=True, subplots=True)
    target_df.plot(y=[ 'shape'], ax=axes[0], color = color_1, alpha=0.3, use_index=True, subplots=True)
    target_df.plot(y=[ 'pattern_n'], ax=axes[0], color = color_1, alpha=1, use_index=True, subplots=True)
    
    first_legend = axes[0].legend(loc ='best', fancybox=True ,framealpha=0.1)
    axes[0].add_artist(first_legend)
    
    axes[0].set_xlabel('index')
    #axes[0].axes.xaxis.set_visible(False)
    axes[0].set_ylabel(unit)
    #axes[0].set_title('{} [{}] - Euclidean Dist. Threshold = {}'.format(param,shape_type, threshold))
    
    # Second Subplot:
    
    target_df.plot(y=[ 'dist'], ax=axes[1], color = [ 'grey'], alpha=1, use_index=True, subplots=True, sharex=True)
    second_legend = axes[1].legend(loc ='best', fancybox=True ,framealpha=0.1)
    axes[1].add_artist(second_legend)
    axes[1].set_ylabel('distance')
    axes[1].axhline(y=bottom)
    
    # Page Format;
    plt.subplots_adjust(left=0.1,
                        bottom=0.05, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.1, 
                        hspace=0.1)
    
    axes[1].set_xlim(axes[0].get_xlim())
    #axes[1].set_ylim(axes[0].get_ylim())
    plt.savefig('shape_match',dpi=300, bbox_inches = "tight") 
    fig.tight_layout()
    plt.show()
    
    return shape_df['median'],bottom,target_df['pattern_n'], target_df['shape'], target_df['ed_profile']
####################################################################################

####################################################################################

####################################################################################

# =============================================================================
# pattern = 0
# idx_start = 850
# Temperature_shape_1,Temperature_shape_1_thresh, Temperature_pattern_1, matches_pattern_1,dist_pattern_1 = shape_prototype(data_scaled,idx_start,param,unit,color_list[pattern], percentile)
# 
# percentile = 1
# pattern = 1
# idx_start = 1350
# Temperature_shape_2,Temperature_shape_2_thresh, Temperature_pattern_2, matches_pattern_2,dist_pattern_2 = shape_prototype(data_scaled,idx_start,param,unit,color_list[pattern], percentile)
# ####################################################################################
# percentile = 5
# pattern = 2
# idx_start = 4000
# Temperature_shape_3,Temperature_shape_3_thresh, Temperature_pattern_3, matches_pattern_3,dist_pattern_3 = shape_prototype(data_scaled,idx_start,param,unit,color_list[pattern], percentile)
# ####################################################################################
# ####################################################################################
# 
# pattern = 3
# idx_start = 12000
# Temperature_shape_4,Temperature_shape_4_thresh, Temperature_pattern_4, matches_pattern_4,dist_pattern_4 = shape_prototype(data_scaled,idx_start,param,unit,color_list[pattern], percentile)
# ####################################################################################
# 
# percentile = 7
# pattern = 4
# idx_start = 20150
# Temperature_shape_5,Temperature_shape_5_thresh, Temperature_pattern_5, matches_pattern_5,dist_pattern_5 = shape_prototype(data_scaled,idx_start,param,unit,color_list[pattern], percentile)
# ####################################################################################
# percentile = 8
# pattern = 5
# idx_start = 26150
# Temperature_shape_6,Temperature_shape_6_thresh, Temperature_pattern_6, matches_pattern_6,dist_pattern_6 = shape_prototype(data_scaled,idx_start,param,unit,color_list[pattern], percentile)
# =============================================================================
####################################################################################

# =============================================================================
# pattern = 6
# idx_start = 4050
# Temperature_shape_7,Temperature_shape_7_thresh, Temperature_pattern_7, matches_pattern_7 = shape_prototype(data_scaled,idx_start,param,unit,color_list[pattern], percentile)
# ####################################################################################
# 
# pattern = 7
# idx_start = 5000
# Temperature_shape_8,Temperature_shape_8_thresh, Temperature_pattern_8, matches_pattern_8 = shape_prototype(data_scaled,idx_start,param,unit,color_list[pattern], percentile)
# =============================================================================
####################################################################################

# =============================================================================
# pattern = 8
# idx_start = 100 
# Temperature_shape_9,Temperature_shape_9_thresh, Temperature_pattern_9, matches_pattern_9 = shape_prototype(data_scaled,idx_start,param,unit,color_list[pattern], percentile)
# =============================================================================
####################################################################################



# =============================================================================
# Temperature_medianshape_library = pd.concat([Temperature_shape_1,Temperature_shape_2,Temperature_shape_3,Temperature_shape_4,Temperature_shape_5,Temperature_shape_6], axis=1)
# columns = ['1','2','3','4','5','6']
# Temperature_medianshape_library.columns = columns
# 
# Temperature_pattern_library = pd.concat([Temperature_pattern_1,Temperature_pattern_2,Temperature_pattern_3,Temperature_pattern_4,Temperature_pattern_5,Temperature_pattern_6, data_scaled['Date Time']], axis=1)
# columns = ['1','2','3','4','5','6','Date_Time']
# Temperature_pattern_library.columns = columns
# Temperature_pattern_library = Temperature_pattern_library.set_index(['Date_Time'])
# 
# Temperature_matches_library = pd.concat([matches_pattern_1,matches_pattern_2,matches_pattern_3,matches_pattern_4,matches_pattern_5,matches_pattern_6, data_scaled['Date Time']], axis=1)
# columns = ['1','2','3','4','5','6','Date_Time']
# Temperature_matches_library.columns = columns
# Temperature_matches_library = Temperature_matches_library.set_index(['Date_Time'])
# 
# Temperature_distance_library = pd.concat([dist_pattern_1,dist_pattern_2,dist_pattern_3,dist_pattern_4,dist_pattern_5,dist_pattern_6, data_scaled['Date Time']], axis=1)
# columns = ['1','2','3','4','5','6','Date_Time']
# Temperature_distance_library.columns = columns
# Temperature_distance_library = Temperature_distance_library.set_index(['Date_Time'])
# 
# Temperature_data_scaled = data_scaled.set_index(['Date Time'])
# =============================================================================


