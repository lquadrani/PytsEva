import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme, genpareto
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eva_functions import tsEvaStationary
from eva_functions import tsEvaGetTimeStep
from eva_functions import tsEvaComputeReturnLevelsGEVFromAnalysisObj
from eva_functions import tsEvaComputeReturnLevelsGPDFromAnalysisObj
from eva_functions import tsEvaPlotReturnLevelsGEVFromAnalysisObj
from eva_functions import tsEvaPlotReturnLevelsGPDFromAnalysisObj

# Load the dataset (assuming it's a CSV file, adjust as needed)
# Assuming 'timeAndSeriesHebrides.mat' is a CSV file
current_working_directory = os.getcwd()
data_file_name= current_working_directory+"/test/data/timeAndSeriesHebrides.csv"
data = pd.read_csv(data_file_name, header=None)

# Extract the time and series
timeAndSeries = data.values
minPeakDistanceInDays = 3

print('Stationary fit of extreme value distributions (GEV, GPD) to a time series')


# Stationary fitting (Here, you would fit the GEV and GPD distributions)
# For GEV, we use the generalized extreme value distribution from scipy

statEvaParams=tsEvaStationary(timeAndSeries, minPeakDistanceInDays=minPeakDistanceInDays)

return_periods = [10, 20, 50, 100]

rlevGEV,rlevGEVErr = tsEvaComputeReturnLevelsGEVFromAnalysisObj(statEvaParams, return_periods)
print("rlevGEV=", rlevGEV)
print("rlevGEVErr=", rlevGEVErr)

# Plotting the GEV return levels
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(statEvaParams, 0, ylim=[0.5, 1.5])
plt.title('GEV')
plt.savefig('GEV_ReturnLevels_STATIONARY.png')
plt.show()

# For GPD fitting, we use the generalized Pareto distribution

rlevGPD,rlevGPDErr = tsEvaComputeReturnLevelsGPDFromAnalysisObj(statEvaParams, return_periods)
print("rlevGPD=", rlevGPD)
print("rlevGPDErr=", rlevGPDErr)

# Plotting the GPD return levels
hndl = tsEvaPlotReturnLevelsGPDFromAnalysisObj(statEvaParams, 0, ylim=[0.5, 1.5])
plt.title('GPD')
plt.savefig('GPD_ReturnLevels_STATIONARY.png')
plt.show()

print("Same as before, but the POT is done with a fixed threshold")
potThreshold = np.percentile(timeAndSeries[:, 1], 98)
statEvaParams = tsEvaStationary(timeAndSeries, minPeakDistanceInDays=minPeakDistanceInDays,doSampleData=False,potThreshold=potThreshold)
rlevGPD,rlevGPDErr = tsEvaComputeReturnLevelsGPDFromAnalysisObj(statEvaParams, return_periods)
print ("rlevGPD=",rlevGPD)
print ("rlevGPDErr=",rlevGPDErr)
# Plotting the GPD return levels
hndl = tsEvaPlotReturnLevelsGPDFromAnalysisObj(statEvaParams, 0, ylim=[0.5, 1.5])
plt.title('GPD')
plt.savefig('GPD_ReturnLevels_STATIONARY_THR.png')
plt.show()

print ("Gumbel")
# Gumbel distribution fit (special case of GEV with shape = 0)
statEvaParams = tsEvaStationary(timeAndSeries, minPeakDistanceInDays=minPeakDistanceInDays,gevType='Gumbel')
#print ("GEV statEvaParams=",statEvaParams['GEVstat'])
rlevGEV,rlevGEVErr = tsEvaComputeReturnLevelsGEVFromAnalysisObj(statEvaParams, return_periods)
print("rlevGEV=", rlevGEV)
print("rlevGEVErr=", rlevGEVErr)
# Plotting the GEV return levels
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(statEvaParams, 0, ylim=[.5,1.5])
plt.title('Gumbel')
plt.savefig('Gumbel_ReturnLevels_STATIONARY.png')
plt.show()



