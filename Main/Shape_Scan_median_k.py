
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale


from scipy.stats import iqr
import math





#color_list = ['purple', 'red', 'green', 'yellow', 'pink', 'blue', 'black', 'violet', 'teal', 'brown', 'olive', 'cyan','orange']

# =============================================================================
# shape = conductivity_medianshape_library['1']
# 
# data=conductivity_data_scaled.interpolate(method='pad')
# param='conductivity'
# units='ms/cm'
# color_1 = color_list[pattern]
# threshold = conductivity_shape_1_thresh
# =============================================================================

### LOAD TEMPLATE SHAPE DATA: ##########################################################################
    
def shape_scan(shape,data,param,unit,color_1,threshold):
    #data = param_norm(param,data)
    data = data.reset_index()
    data_target = data
    data = data.interpolate(method='pad')
    param = param
    unit = unit

    
    #NORMALIZE 
    data = data_target
    pattern_length = 24 #hrs
    m = pattern_length*12
    
    shape_data = shape
    

    ###store shapes in data frame 
    
    shape_df = pd.DataFrame(columns=['shape'])
    shape_df['shape'] = shape_data
    
    ### LOAD TARGET DATA: ##########################################################################
    data = data
    #data = data.reset_index()
    data = data.interpolate(method='pad')
    #data['scaled'] = [(num - data_min) / (data_max - data_min) for num in data[param]]
    
    ###store ts and distance profile in data frame
    target_df = pd.DataFrame(columns=[param,'ed_profile','dist','match'])
    target_df[param] = data['scaled']
    
    ed_profile = np.ones(len(data))*np.inf
    for i in range(0, len(data)  - m): #assign and standardize sub_stream from template sequence
        sub_stream = target_df.loc[i:i + m-1 ,param]
        sub_stream=sub_stream.to_numpy()  
        ed_profile[i] = np.linalg.norm(sub_stream - shape_df['shape'].to_numpy()) 
        
    target_df['ed_profile'] = ed_profile
    target_df['dist'] = ed_profile
    target_df['shape'] = np.nan
    
    ### IDENTIFY MATCHES ##########################################################################
    bottom = threshold
    target_df['ed_profile'] = target_df['ed_profile'].mask(target_df['ed_profile'] > bottom)
    condition = target_df['ed_profile'] <= bottom
    ### New TS for TS 1 matches 
    for i in range(0, len(data)-m):
        if target_df.loc[i,'ed_profile']<=bottom: 
            target_df.loc[i:i+m,'shape'] = target_df.loc[i:i+m,param]    

        #i+=m
        #else:
            #target_df.loc[i,'shape'] = np.nan   
    ###########################################################################################
    # Store and Plot shapes 
    
    x = np.arange(0,m+1,1)
    i=0
    j=0
    
    matches = sum(condition)
    matches = pd.DataFrame(index=np.arange(0,m,1), columns=np.arange(matches))
    match_idx = []
    
    length = sum(condition)
    
    while i < len(condition):
        #print(i)
        if math.isnan(target_df.loc[i,'ed_profile'])==False: 

            matches.loc[0:0+m+1,j] = data.loc[i:i+m-1,'scaled']
            match_idx.append(i)
            j += 1 
             
            i += m
        i += 1 
    
    matches = matches.dropna(axis=1, how='all')
    shape_df = shape_df.reset_index()
    shape_df = shape_df.drop(columns=['index'])
    shape_df = pd.concat([shape_df,matches], axis=1)
    
    cols = [col for col in shape_df.columns]

    fig, axes = plt.subplots(nrows=2, ncols=1)

    # First subplot:

    target_df.plot(y=[ param], ax=axes[0], color = [ 'blue'], alpha=1, use_index=True, subplots=True)
    target_df.plot(y=['shape'], ax=axes[0], color = color_1, alpha=1, use_index=True, subplots=True)

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
    
    return match_idx, target_df['shape'], target_df[param], target_df['ed_profile'], target_df['dist']


# =============================================================================
# pattern = 4
# shape = centroids[pattern]
# data = target_data
# param = param
# unit = units
# color_1 = color_list[0]
# threshold = thresh_df.iloc[0,pattern]
# 
# =============================================================================
