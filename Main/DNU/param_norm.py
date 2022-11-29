
import pandas as pd
import numpy as np
from EXO_2022 import *
from EXO_2021 import df_1, df_2, df_3
#from Pressure_Data import df_11,df_12,df_13,df_14
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

from scipy.stats import iqr
import math


def param_norm(param,data):
    if param =='Temperature':
        data['scaled'] = [(num - min_temp) / (max_temp - min_temp) for num in data[param]]
        
    if param =='pH':
        data['scaled'] = [(num - min_pH) / (max_pH - min_pH) for num in data[param]]
            
    if param =='ORP':
        data['scaled'] = [(num - min_orp) / (max_orp - min_orp) for num in data[param]]
            
    if param =='Turbidity':
        data['scaled'] = [(num - min_turbidity) / (max_turbidity - min_turbidity) for num in data[param]]
            
    if param =='fDOM':
        data['scaled'] = [(num - min_fdom) / (max_fdom - min_fdom) for num in data[param]]
        
    if param =='Sp. Conductivity':
        data['scaled'] = [(num - min_cond) / (max_cond - min_cond) for num in data[param]]
        
    if param =='Combined Chlorine':
        data['scaled'] = [(num - min_cl) / (max_cl - min_cl) for num in data[param]]
        
    
    
    return data



