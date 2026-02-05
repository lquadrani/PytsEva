import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva_functions import tsEvaComputeReturnLevelsGEVFromAnalysisObj
from eva_functions import tsEvaNonStationary
from eva_functions import tsEvaPlotSeriesTrendStdDevFromAnalysisObj
from eva_functions import datetime_to_datenum
from eva_functions import tsEvaPlotGEVImageScFromAnalysisObj
from eva_functions import tsEvaPlotReturnLevelsGEVFromAnalysisObj

# A series of TAS (Temperature of Air Surface) yerly maxima is useful to understand how heat waves evolve (Alessandro Dosio, jrc).
# It is a series of yearly maxima, therefore it is fit for a GEV analysis, while a GPD analysis is meaningless.

# Load data
current_working_directory = os.getcwd()
data_file_name= current_working_directory+"/test/data/timeAndSeriesTASMax.csv"
data = pd.read_csv(data_file_name, header=None)
timeAndSeries = data.values
seriesDescr = 'TASMax'

timeWindow = 365.25 * 50  # 50 years
minPeakDistanceInDays = 5*30.2
returnPeriodsInYears = [20, 50, 100, 300]


axisFontSize = 20
axisFontSize3d = 16
labelFontSize = 24
titleFontSize = 26

   
nonStationaryEvaParams, stationaryTransformData, isValid = tsEvaNonStationary( timeAndSeries, timeWindow, minPeakDistanceInDays=minPeakDistanceInDays, extremeLowThreshold=.1, evdType='GEV')

tsEvaPlotSeriesTrendStdDevFromAnalysisObj(nonStationaryEvaParams, stationaryTransformData, ylabel='TAS', legendLocation='upper left')
plt.text(datetime_to_datenum(datetime(2060, 1, 1)), 35, 'Series and trends', fontsize=25)
plt.ylim(0, 40)
plt.savefig('seriesAndTrends_TAS.png')
plt.show()

tsEvaPlotGEVImageScFromAnalysisObj(np.arange(0, 40, 0.001), nonStationaryEvaParams, stationaryTransformData, ylabel='TAS')
plt.text(datetime_to_datenum(datetime(1980, 1, 1)), 35, 'Time varying GEV', fontsize=30)
plt.ylim(0, 40)
plt.savefig('TimeVaryingGEV_TAS.png')
plt.show()

returnLevels, returnLevelsErr = tsEvaComputeReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, returnPeriodsInYears)
print("Return Levels (TAS): ", returnLevels)

timeIndex = 26
rlRange = [0, 14]
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, timeIndex, ylim=rlRange, ylabel='return levels (TAS)')
ax = plt.gca()
ax.set_yticks(range(0, 17, 2))
plt.text(30, 12.5, 'Return level 1995', fontsize=30)
plt.savefig('ReturnLevel1995.png')
plt.show()

timeIndex = len(timeAndSeries) - 4
rlRange = [0, 70]
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, timeIndex, ylim=rlRange, ylabel='return levels (TAS)')
ax = plt.gca()
ax.set_yticks(range(0, 75, 5))
plt.text(30, 65, 'Return level 2095', fontsize=30)
plt.savefig('ReturnLevel2095.png')
plt.show()