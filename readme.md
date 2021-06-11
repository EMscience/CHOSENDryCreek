

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EMscience/CHOSENDryCreek/master)

Please click the above ↑↑ binder link to launch the notebook on cloud.

## Jupyter Supported Interactive Hydrmeteorological Data Preprocessing

Hydrological Data Quality Control and Filling Missing Values

Raw hydrometeorological datasets contain errors, gaps and outliers that needs preprocessing. 
The objective of this work is developing an interactive data preprocessing platform that enables 
acquiring and transforming publicly available raw hydrometeorological data to a ready to use  
dataset. This interactive platform is at the core of the Comprehensive Hydrologic Observatory SEnsor Network CHOSEN dataset (Zhang et al. 2021 submitted to HP). 
[CHOSEN](https://gitlab.com/esdl/chosen) provides a multitude of intensively measured hydrometeorological datasets (e.g., snow melt and soil moisture data besides 
the common precipitation, air temperature and streamflow measurements) across 30 watersheds in the conterminous US. 

### Method

The interactive preprocessing platform starts with acquiring a standard raw hydrometeorological data table and proceeds with cells that perform interactive 
computation to fill missing values. Three missing value filling methods are adopted:

1. Interpolation
2. Regression
3. Climate Caltalog

The details of the methods are described in the notebook file (EM_01_Jupyter Supported Interactive Data Processing Workflow.ipynb).

### File Description

#### Scripts
* EM_01_Jupyter Supported Interactive Data Processing Workflow.ipynb  -- A note book that performs missing value filling using the above three methods. 

	
#### Data

Out of the CHOSEN dataset, the notebook demonstrates the application of the interactive data preprocessing at the [Dry Creek watershed, ID](https://www.boisestate.edu/drycreek/).
1_DryCreek_DischargeTable.csv -- raw streamflow data from the Dry Creek watershed data page.

1_DryCreek_Download_Aggregation.csv -- raw hydro-meteorological data from the Dry Creek watershed data page.


#### Folders
Functions -- contains local routines 

	* Source_QC_Widgets_functions_EM -- holds functions simpliffying multi-staion data filling 

>>
                               =============[********]============== 
\
*Edom Moges* \
*edom.moges@berkeley.edu* \
*[Environmental Systems Dynamics Laboratory (ESDL)](https://www.esdlberkeley.com/)*\
*University of California, Berkeley* 

