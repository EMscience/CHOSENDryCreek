## Dry Creek Notebook
### File Description
#### Data
1_DryCreek_DischargeTable.csv -- raw streamflow data from the Dry Creek watershed data page.

1_DryCreek_Download_Aggregation.csv -- raw hydro-meteorological data from the Dry Creek watershed data page.

#### Scripts
1_DryCreek_Download_Aggregation_Daily.ipynb -- script to aggregate sub-daily data to daily scale.

2_DryCreek_DataFilling.ipynb -- The script the process data filling using the three methods, 
	i)interpolation, ii) regression and iii) climate catalog
	
5_DryCreek_Trim.ipynb -- trims the table to the same timestamp

NetCDF_DryCreek.ipynb -- generates the final netcdf format data

#### Folders
Functions -- contains local routines 

	Abnormal_data_values_control.py -- functions that deals with unrealistic values
	Source_QC_functions_EM.py -- functions simpliffying multi-staion data filling 

#### Presentations
WorkflowChoosen.pptx --- a slide describing the CHOSEN dataset workflow



