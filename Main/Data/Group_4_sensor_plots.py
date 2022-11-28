

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import datetime
import datetime as dt
from datetime import datetime, timedelta

#%% read pickle file as dataframe
df_online_1 = pd.read_pickle('df_online_1.pkl')
df_online_2 = pd.read_pickle('df_online_2.pkl')
df_online_3 = pd.read_pickle('df_online_3.pkl')

df_offline = pd.read_pickle('df_offline.pkl')
pattern_df = pd.read_pickle('pattern_df.pkl')

#%% Plot
if __name__ == "__main__":


    df_online_1.plot(y=['scaled'], color = ['blue'], alpha=0.5, use_index=True, subplots=True)
    plt.savefig('df_online_1.png',dpi=300, bbox_inches = "tight") 
    plt.show()
    
    df_online_2.plot(y=['scaled'], color = ['blue'], alpha=0.5, use_index=True, subplots=True)
    plt.savefig('df_online_2.png',dpi=300, bbox_inches = "tight") 
    plt.show()
    
    df_online_3.plot(y=['scaled'], color = ['blue'], alpha=0.5, use_index=True, subplots=True)
    plt.savefig('df_online_3.png',dpi=300, bbox_inches = "tight") 
    plt.show()
    
    df_offline.plot(y=['scaled'], color = ['orange'], alpha=0.5, use_index=True, subplots=True)
    plt.savefig('df_offline.png',dpi=300, bbox_inches = "tight") 
    plt.show()








