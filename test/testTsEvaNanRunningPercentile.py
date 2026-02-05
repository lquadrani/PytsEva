import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva_functions import tsEvaNanRunningPercentile
from eva_functions import tsEvaRunningMeanTrend
from eva_functions import datenum_to_datetime

# this script tests the function for computing of the running percentile

# Load data
current_working_directory = os.getcwd()
data_file_name= current_working_directory+"/test/data/timeAndSeriesHebrides.csv"
data = pd.read_csv(data_file_name, header=None)
timeAndSeries = data.values
seriesDescr = 'Hebrides'

timeWindow = 365.25 * 6  # 6 years
percent = 80

timeStamps = timeAndSeries[:,0]
series = timeAndSeries[:,1]

axisFontSize = 20
axisFontSize3d = 16
labelFontSize = 24
titleFontSize = 26

   
# Compute running mean trend
trendSeries, filledTimeStamps, filledSeries, nRunMn = tsEvaRunningMeanTrend( timeStamps, series, timeWindow)

# Plotting with legend labels, using datenum_to_datetime for x-axis
filledTimeStamps_dt = np.array([datenum_to_datetime(ts) for ts in filledTimeStamps])
lines = []
labels = []
lines.append(plt.plot(filledTimeStamps_dt, filledSeries, label='Running Mean')[0])
labels.append('Running Mean')

for percent in [80, 90, 95, 98, 99]:
    rnprcnt, err = tsEvaNanRunningPercentile(filledSeries, nRunMn, percent)
    print(f"Error = {err/np.nanmean(rnprcnt)*100} %")
    lines.append(plt.plot(filledTimeStamps_dt, rnprcnt, label=f'{percent}th Percentile')[0])
    labels.append(f'{percent}th Percentile')

plt.xlim(filledTimeStamps_dt.min(), filledTimeStamps_dt.max())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

plt.xlabel('Year', fontsize=labelFontSize)
plt.ylabel('Series', fontsize=labelFontSize)
plt.title('Running Mean Trend', fontsize=titleFontSize)
plt.tick_params(axis='both', labelsize=axisFontSize)
plt.legend(fontsize=axisFontSize, loc='upper right')
plt.show()