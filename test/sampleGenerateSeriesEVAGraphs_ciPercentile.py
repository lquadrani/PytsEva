import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import scipy.io
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eva_functions import tsEvaComputeReturnLevelsGEVFromAnalysisObj
from eva_functions import tsEvaComputeReturnLevelsGPDFromAnalysisObj
from eva_functions import tsEvaNonStationary
from eva_functions import tsEvaPlotSeriesTrendStdDevFromAnalysisObj
from eva_functions import tsEvaPlotGEVImageScFromAnalysisObj
from eva_functions import tsEvaPlotGPDImageScFromAnalysisObj
from eva_functions import tsEvaPlotReturnLevelsGEVFromAnalysisObj
from eva_functions import tsEvaPlotReturnLevelsGPDFromAnalysisObj
from eva_functions import tsEvaPlotTransfToStatFromAnalysisObj
from eva_functions import tsEvaPlotGEV3DFromAnalysisObj
from eva_functions import datetime_to_datenum


# this sample script illustrates how to execute the tsEva to estimate the
# long term variations of the extremes, using a moving percentile to
# estimate the amplitude of the series, instead of the moving standard
# deviation. This approach models better the variations of the extremes
# than the one based on the standard deviation, but is subject to stronger
# uncertainty.

# Load data
current_working_directory = os.getcwd()
data_file_name= current_working_directory+"/test/data/timeAndSeriesHebrides.csv"
data = pd.read_csv(data_file_name, header=None)
timeAndSeries = data.values
extremesRange = [0.2, 1.2]
rlRange = [0.6, 1.1]
seasonalExtrRange = [0.1, 1.1]
seriesDescr = 'Hebrides'

timeWindow = 365.25 * 6  # 6 years
minPeakDistanceInDays = 3
ciPercentile = 98

minTS = np.min(timeAndSeries[:, 0])
maxTS = np.max(timeAndSeries[:, 0])
axisFontSize = 20
axisFontSize3d = 16
labelFontSize = 24
titleFontSize = 26

# Preparing xticks
years = np.arange(1980, 2015, 2)
months = np.ones_like(years)
days = np.ones_like(years)
dtns = np.column_stack((years, months, days))
dts = [datetime(int(y), int(m), int(d)) for y, m, d in zip(years, months, days)]
tickTmStmp = [datetime_to_datenum(dt) for dt in dts]

wr = np.linspace(min(extremesRange), max(extremesRange), 1501)

print('trend only statistics (transformation + eva + backtransformation)')
nonStatEvaParams, statTransfData, isValid = tsEvaNonStationary(timeAndSeries, timeWindow, transfType='trendCIPercentile', ciPercentile=ciPercentile, minPeakDistanceInDays=minPeakDistanceInDays)

print('  plotting the series')
hndl = tsEvaPlotSeriesTrendStdDevFromAnalysisObj(nonStatEvaParams, statTransfData, legendLocation='upper right',ylabel='Lvl (m)', title=seriesDescr, titleFontSize=titleFontSize, dateformat='%y', xtick=tickTmStmp)
print('  saving the series plot')
plt.savefig('seriesTrendOnly_ciPercentile.png')
plt.show()

# Uncomment the following lines if needed
# print('  plotting and saving the 3D GEV graph')
# hndl = tsEvaPlotGEV3DFromAnalysisObj(wr, nonStatEvaParams, statTransfData, xlabel='Lvl (m)', axisfontsize=axisFontSize3d)
# plt.title('GEV 3D', fontsize=titleFontSize)
# plt.savefig('GEV3DTrendOnly_ciPercentile.png')

print('  plotting and saving the 2D GEV graph')
hndl = tsEvaPlotGEVImageScFromAnalysisObj(wr, nonStatEvaParams, statTransfData, ylabel='Lvl (m)', dateformat='%y', xtick=tickTmStmp)
plt.title('GEV', fontsize=titleFontSize)
plt.savefig('GEV2DTrendOnly_ciPercentile.png')
plt.show()

print('  plotting and saving the 2D GPD graph')
hndl = tsEvaPlotGPDImageScFromAnalysisObj(wr, nonStatEvaParams, statTransfData, ylabel='Lvl (m)', dateformat='%y', xtick=tickTmStmp)
plt.title('GPD', fontsize=titleFontSize)
plt.savefig('GPD2DTrendOnly_ciPercentile.png')
plt.show()

#Computing and plotting the return levels for a given time
timeIndex = 999
timeStamps = statTransfData.timeStamps
dtvc = datetime.fromordinal(int(timeStamps[timeIndex]) - 366) # adjusting for matplotlib datenum offset
tmstmpref = datetime(dtvc.year, dtvc.month, 1)
print(f'  plotting return levels for time {tmstmpref.strftime("%d-%b-%Y")}')
print('  ... for GEV the sample is small and the confidence interval is broad')
return_periods = [10, 20, 50, 100]

rlevGEV,rlevGEVErr = tsEvaComputeReturnLevelsGEVFromAnalysisObj(nonStatEvaParams, return_periods,timeIndex=timeIndex)
print("rlevGEV=", rlevGEV)
print("rlevGEVErr=", rlevGEVErr)
hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(nonStatEvaParams, timeIndex, ylim=rlRange)
plt.title('GEV return levels for ' + tmstmpref.strftime('%d-%b-%Y'), fontsize=titleFontSize)
plt.savefig('GEV_ReturnLevels_ciPercentile.png')
plt.show()

rlevGPD,rlevGPDErr = tsEvaComputeReturnLevelsGPDFromAnalysisObj(nonStatEvaParams, return_periods,timeIndex=timeIndex)
print("rlevGPD=", rlevGPD)
print("rlevGPDErr=", rlevGPDErr)
hndl = tsEvaPlotReturnLevelsGPDFromAnalysisObj(nonStatEvaParams, timeIndex, ylim=rlRange)
plt.title('GPD return levels for ' + tmstmpref.strftime('%d-%b-%Y'), fontsize=titleFontSize)
plt.savefig('GPD_ReturnLevels_ciPercentile.png')
plt.show()


print('plotting and saving stationary series')
hndl = tsEvaPlotTransfToStatFromAnalysisObj(nonStatEvaParams, statTransfData, ylabel='Lvl (m)', xlabel='Year', dateformat='%y', xtick=tickTmStmp, ylim=[-4, 11])
plt.savefig('statSeriesTrendOnly_ciPercentile.png')
plt.show()

print('seasonal statistics')
nonStatEvaParams, statTransfData, isValid = tsEvaNonStationary(timeAndSeries, timeWindow, transfType='seasonalCIPercentile', ciPercentile=ciPercentile, minPeakDistanceInDays=minPeakDistanceInDays)

wr = np.linspace(min(seasonalExtrRange), max(seasonalExtrRange), 1501)

print('  plotting a slice of data ')
slice = [1990, 1995]
plotTitle = '1990-1995'

print('    plotting the series')
hndl = tsEvaPlotSeriesTrendStdDevFromAnalysisObj(nonStatEvaParams, statTransfData,ylabel='Lvl (m)', dateformat='%Y', title=plotTitle, minYear=slice[0], maxYear=slice[1], xtick=tickTmStmp)
print('    saving the series plot')
plt.savefig('seriesSeasonal_ciPercentile.png')
plt.show()

print('plotting and saving stationary series')
hndl = tsEvaPlotTransfToStatFromAnalysisObj(nonStatEvaParams, statTransfData,ylabel='Lvl (m)', xlabel='Year', dateformat='%y', minyear=slice[0], maxyear=slice[1], xtick=tickTmStmp)
plt.savefig('statSeriesTrendOnlySeasonal_ciPercentile.png')
plt.show()

print('    plotting and saving the 3D GEV graph')
hndl = tsEvaPlotGEV3DFromAnalysisObj(wr, nonStatEvaParams, statTransfData, xlabel='Lvl (m)', dateformat='%Y', minyear=slice[0], maxyear=slice[1], ytick=tickTmStmp, axisfontsize=axisFontSize3d)
plt.title(f'GEV 3D, {plotTitle}', fontsize=titleFontSize)
plt.savefig('GEV3DSeasonal_ciPercentile.png')
plt.show()

print('    plotting and saving the 2D GEV graph')
hndl = tsEvaPlotGEVImageScFromAnalysisObj(wr, nonStatEvaParams, statTransfData, ylabel='Lvl (m)', minYear=slice[0], maxYear=slice[1], dateformat='%y', xtick=tickTmStmp)
plt.title(f'GEV {plotTitle}', fontsize=titleFontSize)
plt.savefig('GEV2DSeasonal_ciPercentile.png')
plt.show()
print('    plotting and saving the 2D GPD graph')
hndl = tsEvaPlotGPDImageScFromAnalysisObj(wr, nonStatEvaParams, statTransfData, ylabel='Lvl (m)', minYear=slice[0], maxYear=slice[1], dateformat='%y', xtick=tickTmStmp)
plt.title(f'GPD {plotTitle}', fontsize=titleFontSize)
plt.savefig('GPD2DSeasonal_ciPercentile.png')
plt.show()
