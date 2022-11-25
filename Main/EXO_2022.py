

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import datetime
import datetime as dt
from datetime import datetime, timedelta


##### EXO - April May #####################################################################

#GENERAL PRE-PROCESSING/FORMATING/PRELIMINARY OUTLIER REMOVAL:

df_exo_2022= pd.read_csv('EXO_GroupProject_2022.csv',squeeze=True) 

df_exo_2022.loc[0:3313,'Date Time'] = pd.to_datetime(df_exo_2022.loc[0:3313,'Date Time'], utc=True)

df_exo_2022.loc[3314:8497,'Date Time'] = pd.to_datetime(df_exo_2022.loc[3314:8497,'Date Time'], utc=True)

df_exo_2022.loc[8498:11953,'Date Time'] = pd.to_datetime(df_exo_2022.loc[8498:11953,'Date Time'], utc=True)

df_exo_2022.loc[11954:17425,'Date Time'] = pd.to_datetime(df_exo_2022.loc[11954:17425,'Date Time'], utc=True)

df_exo_2022.loc[11954:17425,'Date Time'] = pd.to_datetime(df_exo_2022.loc[11954:17425,'Date Time'], utc=True)

df_exo_2022.loc[17426:20881,'Date Time'] = pd.to_datetime(df_exo_2022.loc[17426:20881,'Date Time'], utc=True)

df_exo_2022.loc[20882:26065,'Date Time'] = pd.to_datetime(df_exo_2022.loc[20882:26065,'Date Time'], utc=True)

df_exo_2022.loc[26066:29521,'Date Time'] = pd.to_datetime(df_exo_2022.loc[26066:29521,'Date Time'], utc=True)

df_exo_2022.loc[29522:34993,'Date Time'] = pd.to_datetime(df_exo_2022.loc[29522:34993,'Date Time'], utc=True)

df_exo_2022.loc[34994:38449,'Date Time'] = pd.to_datetime(df_exo_2022.loc[34994:38449,'Date Time'], utc=True)

df_exo_2022.loc[38450:43921,'Date Time'] = pd.to_datetime(df_exo_2022.loc[38450:43921,'Date Time'], utc=True)

df_exo_2022.loc[43922:47376,'Date Time'] = pd.to_datetime(df_exo_2022.loc[43922:47376,'Date Time'], utc=True)
df_exo_2022.loc[47377:47377,'Date Time'] = pd.to_datetime(df_exo_2022.loc[47377:47377,'Date Time'], utc=True)

df_exo_2022.loc[47378:52561,'Date Time'] = pd.to_datetime(df_exo_2022.loc[47378:52561,'Date Time'], utc=True)

df_exo_2022.loc[52562:56017,'Date Time'] = pd.to_datetime(df_exo_2022.loc[52562:56017,'Date Time'], utc=True)

df_exo_2022.loc[56018:61489,'Date Time'] = pd.to_datetime(df_exo_2022.loc[56018:61489,'Date Time'], utc=True)

df_exo_2022.loc[61490:64945,'Date Time'] = pd.to_datetime(df_exo_2022.loc[61490:64945,'Date Time'], utc=True)

df_exo_2022.loc[64946:,'Date Time'] = pd.to_datetime(df_exo_2022.loc[64946:,'Date Time'], utc=True)


#DEFINE DATA FRAMES 
df_4 = df_exo_2022[2:15020]
df_5 = df_exo_2022[17300:23930]
df_6 = df_exo_2022[25095:]

df_exo_2022 = df_exo_2022[44786:].set_index('Date Time')
#df_5 = df_exo_2022[17300:23930].set_index('Date Time')
#df_6 = df_exo_2022[25095:].set_index('Date Time')

#df_exo_2022 = df_exo_2022.set_index('Date Time')

max_temp = max(df_exo_2022 ['Temperature'])

min_temp = min(df_exo_2022 ['Temperature'])

#ANNOTATE WITH MAINTENENCE ACTIVITIES 
if __name__ == "__main__":

    #axes = df_4.plot(y=['Temperature', 'ORP', 'pH', 'Turbidity', 'Sp. Conductivity', 'fDOM'], color = [ 'orange', 'purple', 'red', 'brown', 'blue', 'green'], alpha=0.5, use_index=True, subplots=True)
    #axes = df_4.plot(y=['Temperature', 'ORP', 'pH', 'Turbidity', 'Sp. Conductivity', 'fDOM'], color = [ 'orange', 'purple', 'red', 'brown', 'blue', 'green'], alpha=0.5, use_index=True, subplots=True)
    
    #axes = df_6.plot(y=['Temperature', 'ORP','pH','Turbidity','Sp. Conductivity', 'fDOM'], color = [ 'orange', 'purple','red', 'brown','blue', 'green'], alpha=0.5, use_index=True, subplots=True)
    axes = df_exo_2022.plot(y=['Temperature'], color = [ 'orange'], alpha=0.5, use_index=True, subplots=True)
# ANNOTATE WITH MAINTENENCE ACTIVITIES: #
    for ax in axes:

        #axes[1].axvline(dt.datetime(2022, 4, 25, 10, 20), color='purple', linestyle = 'dashed', alpha=0.5)
        #axes[5].axvline(dt.datetime(2022, 4, 25, 10, 20), color='purple', linestyle = 'dashed', alpha=0.5)
        #ax.axvline(dt.datetime(2022, 4, 22, 15, 25), color='orange', linestyle = 'dashed')
        #axes[4].axvline(dt.datetime(2022, 5, 4, 10, 45), color='purple', linestyle = 'dashed')
        #axes[3].axvline(dt.datetime(2022, 5, 4, 10, 45), color='purple', linestyle = 'dashed')
        #axes[2].axvline(dt.datetime(2022, 5, 4, 10, 45), color='purple', linestyle = 'dashed')
        #axes[3].axvline(dt.datetime(2022, 5, 5, 12, 10), color='grey', linestyle = 'dashed')
        
        #ax.axvline(dt.datetime(2022, 5, 31, 14, 00), color='orange', linestyle = 'dashed')
        
        #ax.axvline(dt.datetime(2022, 6, 1, 12, 00), color='green', linestyle = 'dashed')
        
        #ax.axvline(dt.datetime(2022, 6, 27, 15,5), color='green', linestyle = 'dashed')
        
        #ax.axvline(dt.datetime(2022, 7, 12, 11,3), color='orange', linestyle = 'dashed')
        
        #ax.axvline(dt.datetime(2022, 8, 5, 8,3), color='orange', linestyle = 'dashed')

                
        axes[0].set_ylabel("Deg. C")
        #axes[1].set_ylabel("mv")
        #axes[2].set_ylabel("pH")
        #axes[3].set_ylabel("NTU")
        #axes[4].set_ylabel("ms/cm")
        #axes[5].set_ylabel("QSU")
        
        axes[0].legend(bbox_to_anchor=(1.0, .9))
        #axes[1].legend(bbox_to_anchor=(1.2, 1.0))
        #axes[2].legend(bbox_to_anchor=(1.0, 1.0))
        #axes[3].legend(bbox_to_anchor=(1.0, 1.1))
        plt.savefig('EXO_plt.png',dpi=300, bbox_inches = "tight") 

#ANNOTATE WITH MAINTENENCE ACTIVITIES 
if __name__ == "__main__":

    #axes = df_4.plot(y=['Temperature', 'ORP', 'pH', 'Turbidity', 'Sp. Conductivity', 'fDOM'], color = [ 'orange', 'purple', 'red', 'brown', 'blue', 'green'], alpha=0.5, use_index=True, subplots=True)
    #axes = df_4.plot(y=['Temperature', 'ORP', 'pH', 'Turbidity', 'Sp. Conductivity', 'fDOM'], color = [ 'orange', 'purple', 'red', 'brown', 'blue', 'green'], alpha=0.5, use_index=True, subplots=True)
    
    #axes = df_6.plot(y=['Temperature', 'ORP','pH','Turbidity','Sp. Conductivity', 'fDOM'], color = [ 'orange', 'purple','red', 'brown','blue', 'green'], alpha=0.5, use_index=True, subplots=True)
    #axes = df_5.plot(y=['Temperature'], color = [ 'blue'], alpha=0.5, use_index=True, subplots=True)
# ANNOTATE WITH MAINTENENCE ACTIVITIES: #
    #for ax in axes:

        #axes[1].axvline(dt.datetime(2022, 4, 25, 10, 20), color='purple', linestyle = 'dashed', alpha=0.5)
        #axes[5].axvline(dt.datetime(2022, 4, 25, 10, 20), color='purple', linestyle = 'dashed', alpha=0.5)
        #ax.axvline(dt.datetime(2022, 4, 22, 15, 25), color='orange', linestyle = 'dashed')
        #axes[4].axvline(dt.datetime(2022, 5, 4, 10, 45), color='purple', linestyle = 'dashed')
        #axes[3].axvline(dt.datetime(2022, 5, 4, 10, 45), color='purple', linestyle = 'dashed')
        #axes[2].axvline(dt.datetime(2022, 5, 4, 10, 45), color='purple', linestyle = 'dashed')
        #axes[3].axvline(dt.datetime(2022, 5, 5, 12, 10), color='grey', linestyle = 'dashed')
        
        #ax.axvline(dt.datetime(2022, 5, 31, 14, 00), color='orange', linestyle = 'dashed')
        
        #ax.axvline(dt.datetime(2022, 6, 1, 12, 00), color='green', linestyle = 'dashed')
        
        #ax.axvline(dt.datetime(2022, 6, 27, 15,5), color='green', linestyle = 'dashed')
        
        #ax.axvline(dt.datetime(2022, 7, 12, 11,3), color='orange', linestyle = 'dashed')
        
        #ax.axvline(dt.datetime(2022, 8, 5, 8,3), color='orange', linestyle = 'dashed')

                
        #axes[0].set_ylabel("Deg. C")
        #axes[1].set_ylabel("mv")
        #axes[2].set_ylabel("pH")
        #axes[3].set_ylabel("NTU")
        #axes[4].set_ylabel("ms/cm")
        #axes[5].set_ylabel("QSU")
        
        #axes[0].legend(bbox_to_anchor=(1.0, .9))
        #axes[1].legend(bbox_to_anchor=(1.2, 1.0))
        #axes[2].legend(bbox_to_anchor=(1.0, 1.0))
        #axes[3].legend(bbox_to_anchor=(1.0, 1.1))


    
    plt.savefig('EXO_plt.png',dpi=300, bbox_inches = "tight") 
    plt.show()







