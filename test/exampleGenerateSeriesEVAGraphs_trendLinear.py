"""
This sample script illustrates how to execute tsEva to estimate the long-term
variations of the extremes using a linear trend to model the amplitude of the
series (transfType='trendlinear').  The amplitude is estimated via a running
percentile rather than a running standard deviation, which typically models
the extremes better but comes with stronger uncertainty.

Python port of exampleGenerateSeriesEVAGraphs_trendLinear.m
"""

import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for testing
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsEva import (
    tsEvaNonStationary,
    tsEvaPlotSeriesTrendStdDevFromAnalysisObj,
    tsEvaPlotGEVImageScFromAnalysisObj,
    tsEvaPlotGPDImageScFromAnalysisObj,
    tsEvaPlotReturnLevelsGEVFromAnalysisObj,
    tsEvaPlotReturnLevelsGPDFromAnalysisObj,
    tsEvaPlotTransfToStatFromAnalysisObj,
    tsEvaComputeReturnLevelsGEVFromAnalysisObj,
    tsEvaComputeReturnLevelsGPDFromAnalysisObj,
    tsPlotSeriesPotGPDRetLevFromAnalysisObj,
    tsPlotSeriesYearMaxGEVRetLevFromAnalysisObj,
    datetime_to_datenum,
)

# ---------------------------------------------------------------------------
# Load data — Adriatic TWL from EOatSEE.mat
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
mat = scipy.io.loadmat(os.path.join(script_dir, "data", "EOatSEE.mat"))
tm  = mat['tm'].flatten()
twl = mat['twl'].flatten()
timeAndSeries = np.column_stack((tm, twl))

seriesDescr = 'Adriatic TWL'
extremesRange = [0.5, 2.0]
rlRange = [0.5, 2.5]

timeWindow = 365.25 * 15  # 15 years
minPeakDistanceInDays = 14
ciPercentile = 99

axisFontSize = 20
labelFontSize = 24
titleFontSize = 26

# Preparing xticks (1994–2022, every 2 years)
years = np.arange(1994, 2023, 2)
dts = [datetime(int(y), 1, 1) for y in years]
tickTmStmp = [datetime_to_datenum(dt) for dt in dts]

wr = np.linspace(min(extremesRange), max(extremesRange), 1501)

# ---------------------------------------------------------------------------
# Run non-stationary EVA with trendlinear transformation
# ---------------------------------------------------------------------------
print('trend linear statistics (transformation + eva + backtransformation)')
nonStatEvaParams, statTransfData, isValid = tsEvaNonStationary(
    timeAndSeries, timeWindow,
    transfType='trendlinear',
    ciPercentile=ciPercentile,
    potPercentiles=list(np.arange(97, 99.5, 0.5)),  # [97, 97.5, 98, 98.5, 99] — matches MATLAB
    minPeakDistanceInDays=minPeakDistanceInDays,
)

if not isValid:
    print("EVA failed — isValid=False")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Series plot
# ---------------------------------------------------------------------------
print('  plotting the series')
hndl = tsEvaPlotSeriesTrendStdDevFromAnalysisObj(
    nonStatEvaParams, statTransfData,
    legendLocation='upper right',
    ylabel='TWL (m)',
    title=seriesDescr,
    titleFontSize=titleFontSize,
    dateformat='%y',
    xtick=tickTmStmp,
)
plt.savefig('seriesTrendLinear.png')
plt.show()

# ---------------------------------------------------------------------------
# Series + POT peaks + time-varying GPD return levels
# ---------------------------------------------------------------------------
print('  plotting POT peaks and GPD return levels on original series')
hndl = tsPlotSeriesPotGPDRetLevFromAnalysisObj(
    nonStatEvaParams, statTransfData,
    ylabel='TWL (m)', dateformat='%y', xtick=tickTmStmp,
)
plt.title('GPD return levels', fontsize=titleFontSize)
plt.savefig('PotAndReturnLevelsTrendLinear.png')
plt.show()

# ---------------------------------------------------------------------------
# Series + annual maxima + time-varying GEV return levels
# ---------------------------------------------------------------------------
print('  plotting annual maxima and GEV return levels on original series')
hndl = tsPlotSeriesYearMaxGEVRetLevFromAnalysisObj(
    nonStatEvaParams, statTransfData,
    ylabel='TWL (m)', dateformat='%y', xtick=tickTmStmp,
)
plt.title('GEV return levels', fontsize=titleFontSize)
plt.savefig('YearMaxGEVReturnLevelsTrendLinear.png')
plt.show()

# ---------------------------------------------------------------------------
# 2D GEV image
# ---------------------------------------------------------------------------
print('  plotting and saving the 2D GEV graph')
hndl = tsEvaPlotGEVImageScFromAnalysisObj(
    wr, nonStatEvaParams, statTransfData,
    ylabel='Lvl (m)', dateformat='%y', xtick=tickTmStmp,
)
plt.title('GEV', fontsize=titleFontSize)
plt.savefig('GEV2DTrendLinear.png')
plt.show()

# ---------------------------------------------------------------------------
# 2D GPD image
# ---------------------------------------------------------------------------
print('  plotting and saving the 2D GPD graph')
hndl = tsEvaPlotGPDImageScFromAnalysisObj(
    wr, nonStatEvaParams, statTransfData,
    ylabel='Lvl (m)', dateformat='%y', xtick=tickTmStmp,
)
plt.title('GPD', fontsize=titleFontSize)
plt.savefig('GPD2DTrendLinear.png')
plt.show()

# ---------------------------------------------------------------------------
# Return levels at two time indices (beginning and end of series)
# ---------------------------------------------------------------------------
timeStamps = statTransfData.timeStamps
return_periods = [5, 10, 30, 100]

for lx, timeIndex in enumerate([999, len(timeStamps) - 1000]):
    if lx == 0:
        ttl = 'GPD return levels for beginning of series'
    else:
        ttl = 'GPD return levels for end of series'

    dtvc = datetime.fromordinal(int(timeStamps[timeIndex]) - 366)
    tmstmpref = datetime(dtvc.year, dtvc.month, 1)
    print(f'  computing return levels for {tmstmpref.strftime("%b-%Y")}')

    rlevGEV, rlevGEVErr = tsEvaComputeReturnLevelsGEVFromAnalysisObj(
        nonStatEvaParams, return_periods, timeIndex=timeIndex)
    hndl = tsEvaPlotReturnLevelsGEVFromAnalysisObj(
        nonStatEvaParams, timeIndex, ylim=rlRange, maxReturnPeriodYears=200)
    plt.title(f'GEV return levels — {tmstmpref.strftime("%b-%Y")}', fontsize=titleFontSize)
    plt.savefig(f'GEV_ReturnLevelsTrendLinear_{"beg" if lx == 0 else "end"}.png')
    plt.show()

    rlevGPD, rlevGPDErr = tsEvaComputeReturnLevelsGPDFromAnalysisObj(
        nonStatEvaParams, return_periods, timeIndex=timeIndex)
    hndl = tsEvaPlotReturnLevelsGPDFromAnalysisObj(
        nonStatEvaParams, timeIndex, ylim=rlRange, maxReturnPeriodYears=200)
    plt.title(ttl, fontsize=titleFontSize)
    plt.savefig(f'GPD_ReturnLevelsTrendLinear_{"beg" if lx == 0 else "end"}.png')
    plt.show()

# ---------------------------------------------------------------------------
# Stationary series plot
# ---------------------------------------------------------------------------
print('plotting and saving stationary series')
hndl = tsEvaPlotTransfToStatFromAnalysisObj(
    nonStatEvaParams, statTransfData,
    ylabel='Lvl (m)', xlabel='Year',
    dateformat='%y', xtick=tickTmStmp,
    ylim=[-4, 11],
)
plt.savefig('statSeriesTrendLinear.png')
plt.show()

print('Done.')
