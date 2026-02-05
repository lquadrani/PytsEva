import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva_functions import tsEvaComputeReturnLevelsGEVFromAnalysisObj
from eva_functions import tsEvaNonStationary
from eva_functions import tsEvaPlotSeriesTrendStdDevFromAnalysisObj

# In a SPI (Standardized Precipitation Index) series peaks are distant at
# least 5 month one.
# It is possible to set the algorithm in order to fit a Gumbel instead of a
# full GEV

# Load data
current_working_directory = os.getcwd()
data_file_name= current_working_directory+"/test/data/timeAndSeries_SPI_179_750E_-16.750N.csv"
data = pd.read_csv(data_file_name, header=None)
timeAndSeries = data.values
seriesDescr = 'SPI_179_750E_-16.750N'

timeWindow = 365.25 * 50  # 50 years
minPeakDistanceInDays = 5*30.2
returnPeriodsInYears = [10, 20, 50, 100]

timeAndSeries[:,1] = -timeAndSeries[:,1]

axisFontSize = 20
axisFontSize3d = 16
labelFontSize = 24
titleFontSize = 26

   
nonStationaryEvaParams, stationaryTransformData, isValid = tsEvaNonStationary( timeAndSeries, timeWindow, minPeakDistanceInDays=minPeakDistanceInDays, transfType='trendCIPercentile', ciPercentile=80, evdType='GEV',gevType='Gumbel')
tsEvaPlotSeriesTrendStdDevFromAnalysisObj(nonStationaryEvaParams, stationaryTransformData, plotPercentile=95., ylabel='-SPI', legendLocation='lower left')

returnLevels, returnLevelsErr = tsEvaComputeReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, returnPeriodsInYears)
returnLevels = returnLevels*-1
print("Return Levels (GEV): ", returnLevels)
plt.savefig('seriesSPI_Gumbel.png')
plt.show()