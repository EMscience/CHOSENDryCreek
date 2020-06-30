
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import copy
plt.rcParams.update({'figure.max_open_warning': 0})

def threC(prep,varname):
# This function returns a dataframe without unrealistic values
# This function only delete unrealistic data, doesn't deal with the outliers
# SR avg is 1361 W/m2 for earth, could be net solar radiation so include negative values
    
    thre_dic = {'Discharge':(0,9999999), 'Precipitation':(0,9999), 'AirTemperature':(-20,50),
                'SolarRadiation':(-100,400), 'RelativeHumidity': (0,100), 'WindDirection': (0,365),
                'WindSpeed': (0, 99), 'SWE': (0,9999), 'SnowDepth': (0,9999), 
                'VaporPressure': (0,101), 'SoilMoisture': (0,100), 'SoilTemperature': (-20,50),
                'Isotope': (-9999,9999), 'DewPointTemperature': (-100,100), 'Snowmelt': (0,9999)
                 }
    thmin = thre_dic[varname][0]
    thmax = thre_dic[varname][1]
    
    for i in range(len(prep.columns)):
        col = prep.columns[i]
      
        # tranfer the values to be numeric 
        prep[col] = np.array(pd.to_numeric(prep[col], errors='coerce'))
        
        prep.iloc[np.ravel(np.argwhere(np.array(prep[col]) < thmin)),i] = np.nan
        if np.ravel(np.argwhere(np.array(prep[col]) < thmin)).shape[0]!=0:     
            print(col,'Out of lower threshold indexes:', np.ravel(np.argwhere(np.array(prep[col]) < thmin)))
            print(col,'Out of lower threshold number:', np.ravel(np.argwhere(np.array(prep[col]) < thmin)).shape[0])        

        prep.iloc[np.ravel(np.argwhere(np.array(prep[col]) > thmax)),i] = np.nan
        if np.ravel(np.argwhere(np.array(prep[col]) >thmax)).shape[0]!=0:
            print(col,'Out of upper threshold indexes:', np.ravel(np.argwhere(np.array(prep[col]) > thmax)))
            print(col,'Out of upper threshold number:', np.ravel(np.argwhere(np.array(prep[col]) > thmax)).shape[0])        

        # plt.figure(i)
        # prep_new[prep_new.columns[i]].plot()
    
    return prep

## the function below is used to check the interpolation, regression and climate catalog values
## The threshold needs to bechanged
def outvalues(table_orig, table_fill, table_flag, col_name):
    prep = copy.deepcopy(table_orig[[col_name]])
    #fill_df = copy.deepcopy(table_fill)
    #flag_df = copy.deepcopy(table_flag)
    var = col_name.split('_')[1]
    Variable_max = prep.max().values[0] ### 
    Variable_min = prep.min().values[0]

    thre_dic = {'Discharge':(0,Variable_max), 'Precipitation':(0,Variable_max), 'AirTemperature':(Variable_min,Variable_max),
                'SolarRadiation':(Variable_min,Variable_max), 'RelativeHumidity': (0,Variable_max), 'WindDirection': (0,365),
                'WindSpeed': (0, Variable_max), 'SWE': (0,Variable_max), 'SnowDepth': (0,Variable_max), 
                'VaporPressure': (0,Variable_max), 'SoilMoisture': (0,Variable_max), 
                'SoilTemperature': (Variable_min,Variable_max),
                'Isotope': (Variable_min,Variable_max), 
                'DewPointTemperature': (Variable_min,Variable_max), 
                'Snowmelt': (0,Variable_max)
                 }
    
    k = 0
    for i in np.arange(len(prep)):
        if table_fill.loc[prep.index[i],col_name] >  thre_dic[var][1]:           
            k = k + 1
            print(col_name,'More than maximum, value',table_fill.loc[prep.index[i],col_name],'>',thre_dic[var][1],'index = ',i)
            table_fill.loc[prep.index[i], col_name] = None # set the values back to None
            table_flag.loc[prep.index[i], col_name] = int(0)
            
        elif table_fill.loc[prep.index[i],col_name] < thre_dic[var][0]:
            k = k + 1
            print(col_name,'Less than minimum, value',table_fill.loc[prep.index[i],col_name],'<',thre_dic[var][0],'index = ',i)
            table_fill.loc[prep.index[i], col_name] = None # set the values back to None
            table_flag.loc[prep.index[i], col_name] = int(0)
    if k == 0:
        print("No filled values are out of original data range")
     