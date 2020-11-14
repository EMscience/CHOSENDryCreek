#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import copy
from pandas.plotting import register_matplotlib_converters
from sklearn import linear_model
from sklearn.metrics import r2_score
import copy
import matplotlib.backends.backend_pdf
lm = linear_model.LinearRegression()
np.random.seed(50)
plt.rcParams.update({'figure.max_open_warning': 0})
register_matplotlib_converters()


def Date_to_float(dt_time): # Converts dates to float by keeping the year
    return dt_time.year + (30*(dt_time.month-1) + dt_time.day)/367


def widgetInterpolation(table, QOI, interplimit, Intpmethod, Intplimit_direction,
                        xmin=None, xmax=None, ymin=None, ymax=None):
    
    y = table[QOI]
     
    ind = np.where(~np.isnan(y)) # index where the value of y1 are not NaN
    lenNonNan = np.shape(ind)[1]  # number of non nan data points 
    totLen = np.shape(y)[0] # record length
    
    print('Total Record Length = ',totLen , ', Length of non NAN =', lenNonNan, '\n')
    
    if np.shape(y)[0] > lenNonNan:
        print('There are ', totLen - lenNonNan, ' missing values.', '\n' )
    yIntp = y.interpolate(limit=interplimit, method=Intpmethod,limit_direction=Intplimit_direction)
    
    # location where interpolation happened
   
    ind_interp = np.argwhere( np.isnan(y[:].to_numpy())!=  np.isnan(yIntp.to_numpy())  )
      
    indAfterInter = np.where(~np.isnan(yIntp)) # index where the value of y1 are not NaN - after interpolation
    lenNonNanAfter = np.shape(indAfterInter)[1]  # number of non nan data points after interpolation
    
    print('Length of interpolated values = ', lenNonNanAfter - lenNonNan, '\n')
    
    # Evaluate whether the interpolated values are out of the original range
    
    if np.max(yIntp) > np.max(y) or np.min(yIntp) < np.min(y):
        print("Interpolated values are out of the original data range.")
    else:
        print("No Interpolated values are out of the original data range.")
    
    return y,yIntp

# poool a variable from the input table based on QOI split -> into new table
# Perform regression for the QOI column

def widgetRegression(table, QOI, RegThreshold,xmin=None, xmax=None, ymin=None, ymax=None):
    
    
    col_names = pd.Series(table.columns)
    varName = QOI.split('_')[1]

    df = copy.deepcopy(table[QOI].to_frame())
    df2 = copy.deepcopy(table[QOI].to_frame())
    df2[QOI] = np.nan
    
    
    ind = np.where(~np.isnan(table[QOI])) # index where the value of y1 are not NaN
    lenNonNan = np.shape(ind)[1]  # number of non nan data points 
    totLen = np.shape(table[QOI])[0] # record length
    
    print('Total Record Length = ',totLen , ', Length of non NAN =', lenNonNan, '\n')
    if np.shape(table[QOI])[0] > lenNonNan:
        print('There are ', totLen - lenNonNan, ' missing values.', '\n' )
    
    # print(np.shape(df.index))
    for i in range(len(col_names)):
        if QOI.split('_')[1] == col_names[i].split('_')[1]:
            #print(i, np.shape(table1[table1.columns[i]]), np.shape(df[QOI]))
            df.loc[df.index,col_names[i]] = table.loc[df.index,col_names[i]]

    print('Number of potential predictors = ',len(df.columns)-1,'\n')
    regR = regressorFunc(df,regThres=RegThreshold)
    # Location where regression is perfored
    ind_regression = np.argwhere(regR.reshape(len(regR)) != df.iloc[:,0].to_numpy())
    
    #print('Total record length =',np.shape(df)[0],'\n')
    print('Number of regressions performed =',len(ind_regression),'\n')
    
    if np.max(regR) > np.max(df[QOI]) or np.min(regR) < np.min(df[QOI]):
        print("Regression values are out of the original data range.")
    else:
        print("No Regression values are out of the original data range.")
          
    df2[QOI] = regR
    
    return df[QOI],df2[QOI]



def regressorFunc(table2, regThres):
    # table2 is the input where:
    # column 1 is y - predicted
    # the remaining columns are x - predictors.
    # regThres - threshold a site to be used for regression
    
    # ===================================
    # separate x predictor and y predicted
    
    # prepare y for the model
    # identify NAN locations for regression
    ind = np.argwhere(~np.isnan(table2.iloc[:,0].to_numpy())) # non - nan values in y
    ind2 = np.argwhere(np.isnan(table2.iloc[:,0].to_numpy())) # nan values in y
    dim = ind.shape[0]
    #print(dim,ind2.shape[0])

    # y without NAN, where model is to be developed.
    yx = copy.deepcopy(table2.iloc[:,0][ind[:,0]]) # non nan values
    yx = copy.deepcopy(np.array([yx]))
    
    entireY = copy.deepcopy( table2.iloc[:,0] )
    entireY = copy.deepcopy(np.array([entireY]))
    
    n = table2.shape[1] # number of columns
    m = table2.shape[0] 
    
    # Repository for correlation coefficient
    y = np.zeros([m,n-1])*np.nan
    r2 = np.zeros([n-1])

 
    for i in np.arange(1,n):
        entireX = copy.deepcopy( table2.iloc[:,i] )
        entireX = copy.deepcopy(np.array([entireX]))
        
        x1 = copy.deepcopy( table2.iloc[:,i][ind[:,0]] )
        x1 = copy.deepcopy(np.array([x1]))
        
        
        #=================================
        # Fit model and calculate r2
        # Fit the model, where both x and y are not nan
        
        x = x1
        if np.isnan(x).all() == False: # Liang edited, make sure there is some value in x
            bad = ~np.logical_or(np.isnan(yx), np.isnan(x)) # non NAN index in both variables
    
            dim = len(np.argwhere(bad[0,:]==True))
            qw1 = np.compress(bad[0,:], yx).reshape(dim,1) # station
            qw2 = np.compress(bad[0,:], x).reshape(dim,1) # potential
            model1 = lm.fit(qw2, qw1) 
            r2[i-1] = r2_score(qw1,model1.predict(qw2))
        
               
            # Use the model where y is missing
            if len(ind2)!=0: # Fit the model when y has NAN
             
                ind3 = np.argwhere(~np.isnan(entireX))[:,1] # Avoid nan in x
                xx = entireX[0,ind3]
                yy = model1.predict(xx.reshape(len(xx.T),1)) # predicted at non NAN in x
                entireX[0,ind3] = yy.reshape(len(yy)) # back to entire x so that the indexing is straightforward
                y[ind2,i-1] = entireX[0,ind2] # Fill y missing based on x[i]
                #print('i=', i, 'r-square',r2[i-1],len(ind3)) # entireX[0,ind2]
                
        else:  # Liang edited: when there is no data
            r2[i-1] = 0  
            
    # Set the values back into y [predicted y with columns as predicted by each predictor x(i)] based on r2
    sortV = pd.Series(r2)
    rank = sortV.rank(method="min",ascending=False) # starts from max r2 (max to min descending)
    R2_rank = np.array([rank])-1  #rank of each column in y based on r2 [rank 1 becomes rank 0 which has max r2]
    #print(R2_rank.shape)
    
    # Liang edited, because the same rank occurs when there are 0
    if n==1: # only one station for this variable, directly return 
        return entireY.T#,r2, len(ind2)
    
    else:
       
        for j in np.arange(0,int(max(max(R2_rank)))): 
            #print(j)
          
            ind_r = np.argwhere(R2_rank==j) # identify where rank is first and goes to lower
            
            if (ind_r.shape[0]!= 0) and (r2[int(ind_r[0,1])] >= regThres): # perform wherever corr coefficient is > regThres
                #print(ind_r.shape[0,1])
                ### Liang edited
                
                ind4 = np.argwhere(np.isnan(entireY)==True)
                # print(ind_r[0,1],regThres, r2[ind_r[0,1]])
                entireY[0,ind4[:,1]] = y[ind4[:,1],int(ind_r[0,1])]
      

        return entireY.T#,r2, len(ind2)
    
def funcClimateCatalogWg(table,QOI,thrLen=270,corrThr=0.7):
       
    
    ind = np.where(~np.isnan(table[QOI])) # index where the value of y1 are not NaN
    lenNonNan = np.shape(ind)[1]  # number of non nan data points 
    totLen = np.shape(table[QOI])[0] # record length
    
    print('Total Record Length = ',totLen , ', Length of non NAN =', lenNonNan, '\n')
    if np.shape(table[QOI])[0] > lenNonNan:
        print('There are ', totLen - lenNonNan, ' missing values.', '\n' )  


    yearInt = min(table.index.year)
    yearMax = max(table.index.year)+1
    years = np.arange(yearInt,yearMax, 1)


    if table[QOI].min()<0:
        ymin=  table[QOI].min()*1.2
    else:
        ymin=  table[QOI].min()*0.8

    ymax=  table[QOI].max()*1.2
    xmin=0
    xmax=367

    AnnualTable = np.ones([(yearMax-yearInt),xmax])*np.nan

    yinter = {}
    yinter[0]=table[QOI]
    
    #plt.plot(yinter[0])
    
    for year in years:
        
        d = yinter[0][yinter[0].index.year==year].copy(deep=True)
        #print(d.shape[0],year)
        AnnualTable[year-yearInt,0] = year
        if year==yearMax-1:
            #print('T')
            AnnualTable[year-yearInt,1:d.shape[0]+1] = d.values # If the year ends earlier than Dec 31
        else:
            AnnualTable[year-yearInt,(xmax - (d.shape[0]) ):(xmax)] = d.values # If the year does not start at Jan 1


    AnnualMean = (np.nanmean(AnnualTable,axis=0)) #(np.nanmedian(AnnualTable,axis=0))
    AnnualStd = (np.nanstd(AnnualTable,axis=0)) 
    AnnualMax = (np.nanmax(AnnualTable,axis=0)) # for outliers MAx
    AnnualMin = (np.nanmin(AnnualTable,axis=0)) # minimum

    AnnualTable2 = copy.deepcopy(AnnualTable)
    
    
    #======================================================================================
    # Calculate the year with the best/closest correlation coefficient for the nan year.
    # Adopt the values from that year to the NAN values. If the closest year is nan or with corr < 0.7, go to the next non nan year.
    # If this all does not work, Take the mean.
    
    yrs = years
    for yr in yrs:
        corrTable = np.ones([(yearMax-yearInt),2])*np.nan # correlation coefficient table |Year| R2 |
        j = np.argwhere(AnnualTable[:,0]==yr)[0] # the row location of NAN year
        for year in years:
            if year != yr:
                k = np.argwhere(AnnualTable[:,0]==year)[0] # Potential filler year
                bad = ~np.logical_or(np.isnan(AnnualTable[j,1:xmax]), np.isnan(AnnualTable[k,1:xmax])) # NAN index in both years
                qw1 = np.compress(bad[0,:], AnnualTable[j,1:xmax]) #NAN
                qw2 = np.compress(bad[0,:], AnnualTable[k,1:xmax]) #potential
                corrTable[year-yearInt,0] = year
                if len(qw1)>=thrLen or len(qw2)>=thrLen: # if there are more than 9 months of data per year, perform correlation.
                    corrTable[year-yearInt,1] = np.corrcoef(qw1, qw2)[0, 1]
                    #print(yr,year,len(qw1),len(qw2),np.corrcoef(qw1, qw2)[0, 1])


            #=========================================
            #Fill the NAN values by the filler
            #print(yr,all(np.isnan(corrTable[:,1])))
            #if ~np.isnan(np.nanmax(corrTable[:,1]) ): # Skip filling if the entire year is nan
            if all(np.isnan(corrTable[:,1]))==False: # Skip filling if the entire year is nan
                indNAN = np.argwhere(np.isnan(AnnualTable[j,:])) # Location of nan. Wherever true, will be filled.
                indFiller = np.argwhere(corrTable[:,1]==np.nanmax(corrTable[:,1]))[0] # Filler row number
                valFiller = AnnualTable[indFiller,indNAN[:,1]] 

                if np.nanmax(corrTable[:,1]) >= corrThr:         
                    AnnualTable2[j,indNAN[:,1]] = valFiller + np.random.normal(scale=AnnualStd[indNAN[:,1]])# Filler value 
                    

                #==========================================
                # If any remaining, fill it with mean series
                im = np.argwhere(np.isnan(AnnualTable2[j,:])) # When both years have NaN at the same hour.
                AnnualTable2[j,im[:,1]] = AnnualMean[im[:,1]] + np.random.normal(scale=AnnualStd[im[:,1]])# Filled with the mean value.                    


                # Set Outliers caused by random to max and min values
                with np.errstate(invalid='ignore'): # below the minimum
                    ind_outlier = np.argwhere(AnnualTable2[j,1:] < AnnualMin[1:])
                    p = ind_outlier[:,1] 

                    if len(p)!=0:
                        AnnualTable2[j,p+1] = AnnualMin[p+1] # +1 since it scans from 1:
                        #print(j,yr,p,AnnualMin[p])
                with np.errstate(invalid='ignore'):
                    ind_outlier = np.argwhere(AnnualTable2[j,1:] > AnnualMax[1:])
                    p = ind_outlier[:,1]

                    if len(p)!=0:
                        AnnualTable2[j,p+1] = AnnualMax[p+1]
                        #print(p,AnnualMax[np.array(p)],AnnualMax[p])

    ind_Climate ={}


    y5 = yinter[0].copy(deep=True) # yinter is interpolated result

    y5 = y5*np.nan

    for year in years:
        d = y5[y5.index.year==year].copy(deep=True)
        ddim = d.shape[0]
        if year == yearMax-1:
            y5.loc[d.index] = AnnualTable2[year-yearInt,1:d.shape[0]+1] # If the year ends earlier than Dec 31
        else:
            y5.loc[d.index] = AnnualTable2[year-yearInt,(xmax - (d.shape[0]) ):(xmax)]

        y5Temp = copy.deepcopy(y5) 


        # location where Climate look up happened
        ind_Climate[0] = np.argwhere(np.isnan(y5.to_numpy()) != np.isnan(yinter[0].to_numpy()) )
        #print(year,yinter[0].name,len(ind_Climate[0]))
        
        
    print(corrThr,corrTable[:,1])   
    print('Number of days where Climate Catalog is performed =',len(ind_Climate[0]),'\n')
    
    if np.max(y5Temp) > np.max(table[QOI]) or np.min(y5Temp) < np.min(table[QOI]):
        print("Climate Catolog values are out of the original data range.")
    else:
        print("No Climate Catalog values are out of the original data range.")


     
    
    return y5Temp,table[QOI], ind_Climate #AnnualTable2
    
    

    
# In[ ]:


#whereClimateCat.to_csv('ClimateLookupVARMET'), whereINT.to_csv('InterpolationVARMET.csv')

