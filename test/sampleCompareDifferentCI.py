import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import scipy.io
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva_functions import tsEvaNonStationary
from eva_functions import tsEvaPlotGEVImageScFromAnalysisObj
from eva_functions import tsEvaPlotTransfToStatFromAnalysisObj

# This example illustrates the fact that the time-varying amplitude of
# the signal, necessary for the estimation of the extremes using the ts
# approach, can be estimated by means of a moving standard deviation or
# or by means of a moving percentile.
# The approach using the moving percentile models better the variations of the extremes
# than the approach based on the standard deviation, but is subject to stronger
# uncertainty.
 
# Load data
current_working_directory = os.getcwd()
data_file_name= current_working_directory+"/test/data/timeAndSeries_waves_015_220E_055_509N.csv"
#data_file_name= current_working_directory+"/test/data/timeAndSeries_waves_023_688E_059_519N.csv"
data = pd.read_csv(data_file_name, header=None)
timeAndSeries = data.values
seriesDescr = 'waves_015_220E_055_509N'
#sereesDescr = 'waves_023_688E_059_519N'

timeWindow = 365.25 * 30  # 30 years
minPeakDistanceInDays = 3
ylabel = 'Hs (m)'

axisFontSize = 20
axisFontSize3d = 16
labelFontSize = 24
titleFontSize = 26

print('trying moving standard deviation ...')
nonStatEvaParams, statTransfData, _ = tsEvaNonStationary(timeAndSeries, timeWindow, minPeakDistanceInDays=minPeakDistanceInDays)
minext = (max(statTransfData.nonStatSeries) + 3*min(statTransfData.nonStatSeries))/4
maxext = max(statTransfData.nonStatSeries)*1.2
xext = np.arange(minext, maxext, 0.01)
print('  plotting time varying GEV')

tsEvaPlotGEVImageScFromAnalysisObj(xext, nonStatEvaParams, statTransfData, ylabel=ylabel,axisFontSize=axisFontSize, axisFontSize3d=axisFontSize3d, labelFontSize=labelFontSize, titleFontSize=titleFontSize)
plt.title('standard deviation', fontsize=titleFontSize)
plt.savefig('DifferentCI_stdDev.png')
plt.show()

print('  transformation diagnostic plot')
tsEvaPlotTransfToStatFromAnalysisObj(nonStatEvaParams, statTransfData, ylabel=ylabel, xlabel='Year',axisFontSize=axisFontSize, labelFontSize=labelFontSize, titleFontSize=titleFontSize)
plt.title('standard deviation', fontsize=titleFontSize)
plt.savefig('DifferentCI_stdDev_Diagnostic.png')
plt.show()

print('trying moving 98th percentile ...')
nonStatEvaParams, statTransfData, _ = tsEvaNonStationary(timeAndSeries, timeWindow, transfType='trendCIPercentile', ciPercentile=98, minPeakDistanceInDays=minPeakDistanceInDays)
print('  plotting time varying GEV')
tsEvaPlotGEVImageScFromAnalysisObj(xext, nonStatEvaParams, statTransfData, ylabel=ylabel,axisFontSize=axisFontSize, axisFontSize3d=axisFontSize3d, labelFontSize=labelFontSize, titleFontSize=titleFontSize)
plt.title('98th percentile', fontsize=titleFontSize)
plt.savefig('DifferentCI_98thPercentile.png')
plt.show()

print('  transformation diagnostic plot')
tsEvaPlotTransfToStatFromAnalysisObj(nonStatEvaParams, statTransfData, ylabel=ylabel, xlabel='Year', axisFontSize=axisFontSize, labelFontSize=labelFontSize, titleFontSize=titleFontSize)
plt.title('98th percentile', fontsize=titleFontSize)
plt.savefig('DifferentCI_98thPercentile_Diagnostic.png')
plt.show()

print('trying moving 98.5th percentile ...')
nonStatEvaParams, statTransfData, _ = tsEvaNonStationary(timeAndSeries, timeWindow, transfType='trendCIPercentile', ciPercentile=98.5, minPeakDistanceInDays=minPeakDistanceInDays)
print('  plotting time varying GEV')
tsEvaPlotGEVImageScFromAnalysisObj(xext, nonStatEvaParams, statTransfData, ylabel=ylabel,axisFontSize=axisFontSize, axisFontSize3d=axisFontSize3d, labelFontSize=labelFontSize, titleFontSize=titleFontSize)
plt.title('98.5th percentile', fontsize=titleFontSize)
plt.savefig('DifferentCI_98.5thPercentile.png')
plt.show()

print('  transformation diagnostic plot')
tsEvaPlotTransfToStatFromAnalysisObj(nonStatEvaParams, statTransfData, ylabel=ylabel, xlabel='Year', axisFontSize=axisFontSize, labelFontSize=labelFontSize, titleFontSize=titleFontSize)
plt.title('98.5th percentile', fontsize=titleFontSize)
plt.savefig('DifferentCI_98.5thPercentile_Diagnostic.png')
plt.show()

print('trying moving 99th percentile ...')
nonStatEvaParams, statTransfData, _ = tsEvaNonStationary(timeAndSeries, timeWindow, transfType='trendCIPercentile', ciPercentile=99, minPeakDistanceInDays=minPeakDistanceInDays)
print('  plotting time varying GEV')
tsEvaPlotGEVImageScFromAnalysisObj(xext, nonStatEvaParams, statTransfData, ylabel=ylabel,axisFontSize=axisFontSize, axisFontSize3d=axisFontSize3d, labelFontSize=labelFontSize, titleFontSize=titleFontSize)
plt.title('99th percentile', fontsize=titleFontSize)
plt.savefig('DifferentCI_99thPercentile.png')
plt.show()

print('  transformation diagnostic plot')
tsEvaPlotTransfToStatFromAnalysisObj(nonStatEvaParams, statTransfData, ylabel=ylabel, xlabel='Year', axisFontSize=axisFontSize, labelFontSize=labelFontSize, titleFontSize=titleFontSize)
plt.title('99th percentile', fontsize=titleFontSize)
plt.savefig('DifferentCI_99thPercentile_Diagnostic.png')
plt.show()
