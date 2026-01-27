import numpy as np
import pandas as pd
import matplotlib.cm as cm
from scipy.signal import find_peaks
from scipy.stats import norm, gumbel_r, genpareto
from scipy.stats import genextreme as gev
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.dates as mdates
from datetime import datetime, timedelta


class Bootstrap_fit:
    n_bootstraps = 200
    
    def __init__(self, data):
        self.data = data

    def fit_genextreme(self):
        # List to store bootstrap parameters
        paramEsts=[]
        bootstrap_params = []
        percentiles = [32, 68]  # Desired percentiles (min, max)
        paramEsts=gev.fit(self.data)

        for j in range(self.n_bootstraps):
            # Resample with replacement
            sample = np.random.choice(self.data, size=len(self.data), replace=True)
            
            try:
                # Fit the GEV distribution to the sample using MLE method
                params_bs = gev.fit(sample, method="MLE", loc=paramEsts[1], scale=paramEsts[2])
                # Append the parameters from this bootstrap sample
                bootstrap_params.append(params_bs)
            except Exception as e:
                # In case the fit fails for a bootstrap sample, skip this sample
                continue

        # Convert the list to a NumPy array
        bootstrap_params = np.array(bootstrap_params)
        bootstrap_params[:,0] = -bootstrap_params[:,0]  # Change sign of shape parameter to match GEV definition
        
        ci_lower = np.percentile(bootstrap_params, percentiles[0],axis=0)
        ci_upper = np.percentile(bootstrap_params, percentiles[1],axis=0)
        paramCIs = np.vstack((ci_lower, ci_upper)).T
        paramEsts=np.mean(bootstrap_params,axis=0)
        return paramEsts, paramCIs

    def fit_gumbel(self):
        # List to store bootstrap parameters
        paramEsts=[]
        bootstrap_params = []
        percentiles = [32, 68]  # Desired percentiles (min, max)
        paramEsts=gumbel_r.fit(self.data)
        
        for j in range(self.n_bootstraps):
            # Resample with replacement
            sample = np.random.choice(self.data, size=len(self.data), replace=True)
            
            try:
                # Fit the GUMBEL distribution
                params_bs = gumbel_r.fit(sample, loc=paramEsts[0], scale=paramEsts[1])
                # Append the parameters from this bootstrap sample
                bootstrap_params.append(params_bs)
            except Exception as e:
                # In case the fit fails for a bootstrap sample, skip this sample
                print(f"Bootstrap sample {j} failed: {e}")
                continue

        # Convert the list to a NumPy array
        bootstrap_params = np.array(bootstrap_params)

        ci_lower = np.percentile(bootstrap_params, percentiles[0],axis=0)
        ci_upper = np.percentile(bootstrap_params, percentiles[1],axis=0)
        paramCIs = np.vstack((ci_lower, ci_upper)).T

        return paramEsts, paramCIs


    def fit_genpareto(self):
        # List to store bootstrap parameters
        paramEsts=[]
        bootstrap_params = []
        percentiles = [32, 68]  # Desired percentiles (min, max)
        paramEsts=genpareto.fit(self.data,method="MLE",loc=np.mean(self.data),scale=np.std(self.data))
        
        for j in range(self.n_bootstraps):
            # Resample with replacement
            sample = np.random.choice(self.data, size=len(self.data), replace=True)
            
            try:
                # Fit the GPD distribution
                params_bs = genpareto.fit(sample,method="MLE",loc=paramEsts[1],scale=paramEsts[2])
                # Append the parameters from this bootstrap sample
                bootstrap_params.append(params_bs)
            except Exception as e:
                # In case the fit fails for a bootstrap sample, skip this sample
                print(f"Bootstrap sample {j} failed: {e}")
                continue

        # Convert the list to a NumPy array
        bootstrap_params = np.array(bootstrap_params)

        ci_lower = np.percentile(bootstrap_params, percentiles[0],axis=0)
        ci_upper = np.percentile(bootstrap_params, percentiles[1],axis=0)
        paramCIs = np.vstack((ci_lower, ci_upper))

        Z_alpha_half = 1.96
        
        # Calculate standard errors from the confidence intervals
        standard_errors = []
        for ci in paramCIs.T:
            # CI = [lower_bound, upper_bound]
            lower_bound, upper_bound = ci
            SE = (upper_bound - lower_bound) / (2 * Z_alpha_half)
            standard_errors.append(SE)

        # Convert to a numpy array for easy access
        standard_errors = np.array(standard_errors)
        paramEsts=np.mean(bootstrap_params,axis=0)
        return paramEsts, paramCIs, standard_errors

  
class ProbObject:
    def __init__(self, subsrs, percent_m, percent, percent_p):
        valid_data = subsrs[~np.isnan(subsrs)]
        self.N = len(valid_data)
        
        # Target values
        self.target_percent = percent
        
        # Percentiles (the actual data values)
        if self.N > 0:
            self.tM = np.nanpercentile(valid_data, percent_m)
            self.t = np.nanpercentile(valid_data, percent)
            self.tP = np.nanpercentile(valid_data, percent_p)
        else:
            self.tM = self.t = self.tP = np.nan

        # Probabilities (normalized 0 to 1)
        self.probM = percent_m / 100.0
        self.prob = percent / 100.0
        self.probP = percent_p / 100.0
        
    @property
    def percentM(self): return self.probM * 100
    
    @property
    def percent(self): return self.prob * 100
    
    @property
    def percentP(self): return self.probP * 100

    def update(self, val, adding=True):
        """Updates internal probabilities when a value enters or leaves the window."""
        if np.isnan(val) or self.N <= (1 if not adding else 0):
            return
        
        n_old = self.N
        n_new = n_old + 1 if adding else n_old - 1
        modifier = 1 if adding else -1
        
        # Logic: count if the new/old value is less than our tracked percentile values
        self.probM = (self.probM * n_old + modifier * (val < self.tM)) / n_new
        self.prob = (self.prob * n_old + modifier * (val < self.t)) / n_new
        self.probP = (self.probP * n_old + modifier * (val < self.tP)) / n_new
        self.N = n_new

    def is_out_of_bounds(self):
        """Check if the target percentile has drifted outside our tracked range."""
        return self.percentM > self.target_percent or self.percentP < self.target_percent

    def interpolate(self):
        """Linearly interpolate to find the value at the target percentage."""
        pM, p, pP = self.percentM, self.percent, self.percentP
        tM, t, tP = self.tM, self.t, self.tP
        target = self.target_percent
        
        if target == pM: return tM
        if pM < target < p:
            h1, h2 = p - target, target - pM
            return (h1 * tM + h2 * t) / (h1 + h2)
        if target == p: return t
        if p < target < pP:
            h1, h2 = pP - target, target - p
            return (h1 * t + h2 * tP) / (h1 + h2)
        if target == pP: return tP
        return np.nan


def tsEvaPlotTransfToStat(timeStamps, statSeries, srsmean, stdDev, thirdMom, fourthMom, **kwargs):
    axisFontSize=kwargs.get('axisFontSize', 20)
    legendFontSize=kwargs.get('legendFontSize', 20)
    xtick=kwargs.get('xtick',[])
    figPosition=kwargs.get('figPosition',[x + 10 for x in [0, 0, 1450, 700]])
    minyear=kwargs.get('minyear',1)
    maxyear=kwargs.get('maxyear',9999)
    dateformat=kwargs.get('dateformat','%Y')
    legendLocation=kwargs.get('legendLocation','upper right')
    ylim=kwargs.get('ylim',None)

    # Update args with passed values
    for key, value in kwargs.items():
        if (key=='axisFontSize'):
            axisFontSize=value
        if (key=='legendFontSize'):
            legendFontSize=value
        if (key=='xtick'):
            xtick=value
        if (key=='figPosition'):
            figPosition=value
        if (key=='minyear'):
            minyear=value
        if (key=='maxyear'):
            maxyear=value
        if (key=='dateformat'):
            dateformat=value
        if (key=='legendLocation'):
            legendLocation=value
        if (key=='ylim'): 
            ylim=value


    min_date=datetime(minyear, 1, 1)
    max_date=datetime(maxyear, 1, 1)
    minTS=min_date.toordinal()
    maxTS=max_date.toordinal()

    filtered_statSeries = statSeries[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    filtered_srsmean = srsmean[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    filtered_stdDev = stdDev[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    filtered_thirdMom = thirdMom[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    filtered_fourthMom = fourthMom[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    filtered_timeStamps = timeStamps[(timeStamps >= minTS) & (timeStamps <= maxTS)]

    minTS = min(filtered_timeStamps);
    maxTS = max(filtered_timeStamps);

    fig, ax = plt.subplots(figsize=(figPosition[2] / 100,figPosition[3] / 100))
    phandles = [fig]
    

    ax.plot(filtered_timeStamps, filtered_statSeries, label='Normalized series', zorder=1)
    ax.plot(filtered_timeStamps, filtered_srsmean, "--", color="k", linewidth=3, label='Mean', zorder=2)
    ax.plot(filtered_timeStamps, filtered_stdDev, "--", color=[0.5, 0, 0], linewidth=3, label='Std. dev.', zorder=2)
    ax.plot(filtered_timeStamps, filtered_thirdMom, color=[0, 0, 0.5], linewidth=3,label='Skewness', zorder=2)
    ax.plot(filtered_timeStamps, filtered_fourthMom, color=[0, 0.4, 0], linewidth=3,label='Kurtosis',zorder=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(dateformat))
    
    ax.legend([ax.lines[0], ax.lines[1], ax.lines[2], ax.lines[3], ax.lines[4]], ['Normalized series', 'Mean', 'Std dev', 'Skewness', 'Kurtosis'],
              fontsize=legendFontSize, loc=legendLocation)
    ax.tick_params(labelsize=axisFontSize)

    if xtick:
        ax.set_xticks(xtick)
        ax.set_xticklabels([datetime.fromordinal(int(t)).strftime(dateformat) for t in xtick])

    ax.set_xlim([minTS,maxTS])
    # Turn grid on
    ax.grid(True)
#    plt.tight_layout()

    if ylim is not None:
        ax.set_ylim(ylim) 
    else:
        all_data = np.concatenate([filtered_statSeries, filtered_srsmean, filtered_stdDev, filtered_thirdMom, filtered_fourthMom])
        data_min = np.nanmin(all_data)
        data_max = np.nanmax(all_data)
        y_margin = 0.1 * (data_max - data_min)
        ax.set_ylim([data_min - y_margin, data_max + y_margin])


    phandles = {
        'fig': fig,
        'ax': ax,
        'normalized_series': ax.lines[0],  # first line
        'mean': ax.lines[1],  # second line plot
        'std_dev': ax.lines[2],  # third line plot
        'skewness': ax.lines[3],  # fourth line plot
        'kurtosis': ax.lines[4],  # fifth line plot
    }
    return phandles

def tsEvaPlotTransfToStatFromAnalysisObj(nonStationaryEvaParams, stationaryTransformData, **kwargs):
    timeStamps = stationaryTransformData.timeStamps
    series = stationaryTransformData.stationarySeries
    srmean = np.zeros_like(series)
    srstddev = np.ones_like(series)
    st3mom = stationaryTransformData.statSer3Mom
    st4mom = stationaryTransformData.statSer4Mom
    
    minyear = kwargs.get('minyear',1)
    maxyear = kwargs.get('maxyear',9999)
    dateFormat = kwargs.get('dateformat','%Y')
    ylim = kwargs.get('ylim',None)

    for key, value in kwargs.items():
        if (key=='minyear'):
            minyear=value
        if (key=='maxyear'):
            maxyear=value
        if (key=='dateformat'):
            dateformat=value
        if (key=='ylim'): 
            ylim=value

    phandles = tsEvaPlotTransfToStat(timeStamps, series, srmean, srstddev, st3mom, st4mom, **kwargs)
    return phandles

def tsEvaPlotGEVImageSc(Y, timeStamps, epsilon, sigma, mu, **kwargs):
    avgYearLength = 365.2425
    nyears = (max(timeStamps) - min(timeStamps)) / avgYearLength
    nelmPerYear = len(timeStamps) / nyears

    # Default arguments
    
    nPlottedTimesByYear=kwargs.get('nPlottedTimesByYear',min(360, round(nelmPerYear)))
    ylabel=kwargs.get('ylabel','levels (m)')
    zlabel=kwargs.get('zlabel','pdf')
    minYear=kwargs.get('minYear',1)
    maxYear=kwargs.get('maxYear',9999)
    dateformat=kwargs.get('dateformat','%Y')
    axisFontSize=kwargs.get('axisFontSize',22)
    labelFontSize=kwargs.get('labelFontSize',28)
    colormap=kwargs.get('colormap', plt.cm.hot_r)
    plotColorbar=kwargs.get('plotColorbar',True)
    figPosition=kwargs.get('figPosition',[x + 10 for x in [0, 0, 1450, 700]])
    xtick=kwargs.get('xtick',[])
    ax=kwargs.get('ax',None)
    
    # Update args with passed values
    for key, value in kwargs.items():
        if (key=='nPlottedTimesByYear'):
            nPlottedTimesByYear=value
        if (key=='ylabel'):
            ylabel=value
        if (key=='zlabel'):
            zlabel=value
        if (key=='minYear'):
            minYear=value
        if (key=='maxYear'):
            maxYear=value
        if (key=='dateformat'):
            dateformat=value
        if (key=='axisFontSize'):
            axisFontSize=value
        if (key=='labelFontSize'):
            labelFontSize=value
        if (key=='colormap'):
            colormap=value
        if (key=='plotColorbar'):
            plotColorbar=value
        if (key=='figPosition'):
            figPosition=value
        if (key=='xtick'):
            xtick=value
        if (key=='ax'):
            ax=value

    min_date=datetime(minYear, 1, 1)
    max_date=datetime(maxYear, 1, 1)
    minTS=min_date.toordinal()
    maxTS=max_date.toordinal()

    sigma = sigma[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    mu = mu[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    timeStamps = timeStamps[(timeStamps >= minTS) & (timeStamps <= maxTS)]


    # Handle figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(figPosition[2]/100, figPosition[3]/100))
        phandles = [fig]
    else:
        phandles = [ax]

    # Setup time range
    L = len(timeStamps)
    minTS = timeStamps[0]
    maxTS = timeStamps[-1]
    
    npdf = int(np.ceil(((maxTS - minTS) / avgYearLength) * nPlottedTimesByYear))
    navg = int(np.ceil(L / npdf))

    plotSLength = npdf * navg
    timeStamps_plot = np.linspace(minTS, maxTS, plotSLength)

    # Handle epsilon values
    if isinstance(epsilon, (list, np.ndarray)):
        if len(epsilon) == 1:
            epsilon0 = np.ones(npdf) * epsilon
    else:
        epsilon_ = np.full(npdf * navg, np.nan)
        epsilon_[:L] = epsilon
        epsilonMtx = epsilon_.reshape(navg, -1)
        epsilon0 = np.nanmean(epsilonMtx, axis=0).T

    # Interpolation for sigma and mu
    sigma_ = np.interp(timeStamps_plot, timeStamps, sigma)
    sigmaMtx = sigma_.reshape(-1, navg)

    if sigmaMtx.shape[0] > 1:
        sigma0 = np.nanmean(sigmaMtx, axis=1)
        sigma0 = sigma0.T
    else:
        sigma0 = np.transpose(sigmaMtx)
    
    mu_ = np.interp(timeStamps_plot, timeStamps, mu)
    muMtx = mu_.reshape(-1, navg)
    if muMtx.shape[0] > 1:
        mu0 = np.nanmean(muMtx, axis=1)
        mu0 = mu0.T  # Transpose
    else:
        mu0 = muMtx.T

    # Create grid for GEV parameters
    X_mesh, epsilonMtx = np.meshgrid(Y, epsilon0)
    _, sigmaMtx = np.meshgrid(Y, sigma0)
    XMtx, muMtx = np.meshgrid(Y, mu0)
    
    # Compute the GEV PDF
    gevvar = gev.pdf(XMtx, c=epsilonMtx, loc=muMtx, scale=sigmaMtx)
    
    # Plotting
    gevvar_transposed = gevvar.T
    
    cax = ax.imshow(gevvar_transposed, aspect='auto', origin='lower', extent=[timeStamps_plot[0], timeStamps_plot[-1], Y[0], Y[-1]])
    phandles.append(cax)


    # Formatting the axes
    ax.set_xlabel('Year', fontsize=labelFontSize)
    ax.set_ylabel(ylabel, fontsize=labelFontSize)
    ax.tick_params(axis='both', labelsize=axisFontSize)

    # Date formatting
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(dateformat))
    ax.grid(True)

    # Adjust xticks if provided
    if xtick:
        ax.set_xticks(xtick)
        ax.set_xticklabels([datetime.fromordinal(int(t)).strftime(dateformat) for t in xtick])
    
    ax.set_xlim([minTS,maxTS])

    # Colorbar
    if plotColorbar:
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(zlabel, fontsize=labelFontSize)

    return phandles

def tsEvaPlotGPDImageSc(Y, timeStamps, epsilon, sigma, threshold, **kwargs):
    avgYearLength = 365.2425
    nyears = (max(timeStamps) - min(timeStamps)) / avgYearLength
    nelmPerYear = len(timeStamps) / nyears

    # Default arguments
    
    nPlottedTimesByYear = kwargs.get('nPlottedTimesByYear',min(360, round(nelmPerYear)))
    ylabel = kwargs.get('ylabel','levels (m)')
    zlabel = kwargs.get('zlabel','pdf')
    minYear = kwargs.get('minYear',1)
    maxYear = kwargs.get('maxYear',9999)
    dateFormat = kwargs.get('dateformat','%Y')
    axisFontSize = kwargs.get('axisFontSize',22)
    colormap=kwargs.get('colormap', plt.cm.hot_r)
    plotColorbar=kwargs.get('plotColorbar',True)
    labelFontSize = kwargs.get('labelFontSize',28)
    figPosition = kwargs.get('figPosition',[x + 10 for x in [0, 0, 1450, 700]])
    xtick = kwargs.get('xtick',[])
    ax=kwargs.get('ax',None)
    
    for key, value in kwargs.items():
        if (key=='nPlottedTimesByYear'):
            nPlottedTimesByYear=value
        if (key=='ylabel'):
            ylabel=value
        if (key=='zlabel'):
            zlabel=value
        if (key=='minYear'):
            minYear=value
        if (key=='maxYear'):
            maxYear=value
        if (key=='dateformat'):
            dateformat=value
        if (key=='axisFontSize'):
            axisFontSize=value
        if (key=='labelFontSize'):
            labelFontSize=value
        if (key=='colormap'):
            colormap=value
        if (key=='plotColorbar'):
            plotColorbar=value
        if (key=='figPosition'):
            figPosition=value
        if (key=='xtick'):
            xtick=value
        if (key=='ax'):
            ax=value

    min_date=datetime(minYear, 1, 1)
    max_date=datetime(maxYear, 1, 1)
    minTS=min_date.toordinal()
    maxTS=max_date.toordinal()

    sigma = sigma[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    threshold = threshold[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    timeStamps = timeStamps[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    
    # Handle figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(figPosition[2]/100, figPosition[3]/100))
        phandles = [fig]
    else:
        phandles = [ax]

    L = len(timeStamps)
    minTS = timeStamps[0]
    maxTS = timeStamps[-1]

    npdf = int(np.ceil(((maxTS - minTS) / avgYearLength) * nPlottedTimesByYear))
    navg = int(np.ceil(L / npdf))

    plotSLength = npdf * navg
    timeStamps_plot = np.linspace(minTS, maxTS, plotSLength)

    # Handle epsilon values
    if isinstance(epsilon, (list, np.ndarray)):
        if len(epsilon) == 1:
            epsilon0 = np.ones(npdf) * epsilon
    else:
        epsilon_ = np.full(npdf * navg, np.nan)
        epsilon_[:L] = epsilon
        epsilonMtx = epsilon_.reshape(navg, -1)
        epsilon0 = np.nanmean(epsilonMtx, axis=0).T

    sigma_ = np.interp(timeStamps_plot, timeStamps, sigma)
    sigmaMtx = sigma_.reshape(-1, navg)

    if sigmaMtx.shape[0] > 1:
        sigma0 = np.nanmean(sigmaMtx, axis=1)
        sigma0 = sigma0.T
    else:
        sigma0 = np.transpose(sigmaMtx)

    threshold_ = np.interp(timeStamps_plot, timeStamps, threshold)
    thresholdMtx = threshold_.reshape(-1, navg)
    if thresholdMtx.shape[0] > 1:
        threshold0 = np.nanmean(thresholdMtx, axis=1)
        threshold0 = threshold0.T
    else:
        threshold0 = thresholdMtx.T
    
    _, epsilonMtx = np.meshgrid(Y, epsilon0)
    _, sigmaMtx = np.meshgrid(Y, sigma0)
    XMtx, thresholdMtx = np.meshgrid(Y, threshold0)
    
    gevvar = genpareto.pdf(XMtx-thresholdMtx, c=epsilonMtx, scale=sigmaMtx)
    
    # Plotting
    gevvar_transposed = gevvar.T
    cax = ax.imshow(gevvar_transposed, aspect='auto', origin='lower',cmap=colormap, extent=[timeStamps_plot[0],timeStamps_plot[-1], Y[0], Y[-1]])
    
    phandles.append(cax)

    # Formatting the axes
    ax.set_xlabel('Year', fontsize=labelFontSize)
    ax.set_ylabel(ylabel, fontsize=labelFontSize)
    ax.tick_params(axis='both', labelsize=axisFontSize)

    # Date formatting
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(dateformat))
    ax.grid(True)

    # Adjust xticks if provided
    if xtick:
        ax.set_xticks(xtick)
        ax.set_xticklabels([datetime.fromordinal(int(t)).strftime(dateformat) for t in xtick])

    ax.set_xlim([minTS,maxTS])  
    
    # Colorbar
    if plotColorbar:
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(zlabel, fontsize=labelFontSize)

    return phandles

def tsEvaPlotGEVImageScFromAnalysisObj(X, nonStationaryEvaParams, stationaryTransformData, **kwargs):
    
    timeStamps = stationaryTransformData.timeStamps
    epsilon = -nonStationaryEvaParams[0]['parameters']['epsilon']
    sigma = nonStationaryEvaParams[0]['parameters']['sigma']
    mu = nonStationaryEvaParams[0]['parameters']['mu']

    phandles = tsEvaPlotGEVImageSc(X, timeStamps, epsilon, sigma, mu, **kwargs)
    
    
    return phandles

def tsEvaPlotGPDImageScFromAnalysisObj(Y, nonStationaryEvaParams, stationaryTransformData, **kwargs):
    timeStamps = stationaryTransformData.timeStamps
    epsilon = nonStationaryEvaParams[1]['parameters']['epsilon']
    sigma = nonStationaryEvaParams[1]['parameters']['sigma']
    threshold = nonStationaryEvaParams[1]['parameters']['threshold']
    phandles = tsEvaPlotGPDImageSc(Y, timeStamps, epsilon, sigma, threshold, **kwargs)
    
    return phandles

def tsEvaPlotSeriesTrendStdDev(timeStamps, series, trend, stdDev, **kwargs):
    confidenceAreaColor = kwargs.get('confidenceAreaColor',np.array([0.741, 0.988, 0.788]))
    confidenceBarColor = kwargs.get('confidenceBarColor',np.array([0.133, 0.545, 0.133]))
    seriesColor = kwargs.get('seriesColor',[1, 0.5, 0.5])
    trendColor = kwargs.get('trendColor','k')
    xlabel = kwargs.get('xlabel','')
    ylabel = kwargs.get('ylabel','level (m)')
    minYear = kwargs.get('minYear',1000)
    maxYear = kwargs.get('maxYear',9999)
    title = kwargs.get('title','')
    axisFontSize = kwargs.get('axisFontSize',22)
    labelFontSize = kwargs.get('labelFontSize',28)
    titleFontSize = kwargs.get('titleFontSize',30)
    legendLocation = kwargs.get('legendLocation','upper left')
    dateformat = kwargs.get('dateformat','%Y')
    figPosition = kwargs.get('figPosition',[10, 10, 1300, 700])
    verticalRange = kwargs.get('verticalRange',None)
    statsTimeStamps = kwargs.get('statsTimeStamps',timeStamps)
    xtick = kwargs.get('xtick',[])
    
    for key, value in kwargs.items():
        if (key=='confidenceAreaColor'): 
            confidenceAreaColor=value
        if (key=='confidenceBarColor'): 
            confidenceBarColor=value
        if (key=='seriesColor'): 
            seriesColor=value
        if (key=='trendColor'): 
            trendColor=value
        if (key=='xlabel'): 
            xlabel=value
        if (key=='ylabel'): 
            ylabel=value
        if (key=='minYear'): 
            minYear=value
        if (key=='maxYear'): 
            maxYear=value
        if (key=='title'): 
            title=value
        if (key=='axisFontSize'): 
            axisFontSize=value
        if (key=='labelFontSize'): 
            labelFontSize=value
        if (key=='titleFontSize'): 
            titleFontSize=value
        if (key=='legendLocation'): 
            legendLocation=value
        if (key=='dateformat'): 
            dateformat=value
        if (key=='figPosition'): 
            figPosition=value
        if (key=='verticalRange'): 
            verticalRange=value
        if (key=='statsTimeStamps'): 
            statsTimeStamps=value
        if (key=='xtick'):
            xtick=value

    # Convert years to matplotlib date numbers for filtering
    min_date=datetime(minYear, 1, 1)
    max_date=datetime(maxYear, 1, 1)
    minTS=min_date.toordinal()
    maxTS=max_date.toordinal()
    
    # Filtering data
    
    filtered_timeStamps = timeStamps[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    filtered_series = series[(timeStamps >= minTS) & (timeStamps <= maxTS)]

    filtered_statsTS = statsTimeStamps[(statsTimeStamps >= minTS) & (statsTimeStamps <= maxTS)]
    
    filtered_trend = trend[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    filtered_stdDev = stdDev[(timeStamps >= minTS) & (timeStamps <= maxTS)]

    upCI = filtered_trend + filtered_stdDev
    downCI = filtered_trend - filtered_stdDev

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(figPosition[2] / 100,figPosition[3] / 100))
    phandles = [fig]

    # Plot series
    line_series, = ax.plot(filtered_timeStamps, filtered_series, color=seriesColor, linewidth=0.5)
    
    phandles.append(line_series)

    # Date formatting on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter(dateformat))

    if xtick:
        ax.set_xticks(xtick)
        ax.set_xticklabels([datetime.fromordinal(int(t)).strftime(dateformat) for t in xtick])

    ax.set_xlim([np.min(filtered_timeStamps), np.max(filtered_timeStamps)])

    # Fill confidence interval area
    xcibar = np.concatenate([filtered_statsTS, filtered_statsTS[::-1]])
    ycibar = np.concatenate([upCI, downCI[::-1]])

    fill_poly = ax.fill(xcibar, ycibar, color=confidenceAreaColor, alpha=0.2, edgecolor='none')
    phandles.append(fill_poly[0])
    
    # Plot trend and confidence bars
    line_trend, = ax.plot(filtered_statsTS, filtered_trend, color=trendColor, linewidth=3)
    phandles.append(line_trend)

    line_upCI, = ax.plot(filtered_statsTS, upCI, color=confidenceBarColor, linewidth=2)
    phandles.append(line_upCI)

    line_downCI, = ax.plot(filtered_statsTS, downCI, color=confidenceBarColor, linewidth=2)
    phandles.append(line_downCI)

    ax.grid(True)

    if verticalRange is not None and len(verticalRange) == 2:
        ax.set_ylim(verticalRange)

    ax.legend([line_series, line_trend, line_downCI], ['Series', 'Trend', 'Std dev'],
              fontsize=labelFontSize, loc=legendLocation)

    ax.tick_params(axis='both', which='major', labelsize=axisFontSize)
    ax.set_xlabel(xlabel, fontsize=labelFontSize)
    ax.set_ylabel(ylabel, fontsize=labelFontSize)

    if title:
        ax.set_title(title, fontsize=titleFontSize)

    fig.tight_layout()

    return phandles

def tsEvaPlotGEV3DFromAnalysisObj(X, nonStationaryEvaParams, stationaryTransformData, **kwargs):
    timeStamps = stationaryTransformData.timeStamps
    epsilon = nonStationaryEvaParams[0]['parameters']['epsilon']
    sigma = nonStationaryEvaParams[0]['parameters']['sigma']
    mu = nonStationaryEvaParams[0]['parameters']['mu']

    phandles = tsEvaPlotGEV3D(X, timeStamps, epsilon, sigma, mu, **kwargs)

    return phandles

def tsEvaPlotGEV3D(X, timeStamps, epsilon, sigma, mu, **kwargs):
    avgYearLength = 365.2425
    # Default arguments
    nPlottedTimesByYear=kwargs.get('nPlottedTimesByYear', 180)
    xlabel=kwargs.get('xlabel','levels (m)')
    ylabel=kwargs.get('ylabel','year')
    zlabel= kwargs.get('zlabel','pdf')
    minyear= kwargs.get('minyear',1)
    maxyear= kwargs.get('maxyear',9999)
    dateformat= kwargs.get('dateformat','%Y')
    axisFontSize= kwargs.get('axisFontSize', 20)
    labelFontSize=kwargs.get('labelFontSize', 20)
    ytick = kwargs.get('ytick',[])

        # Update args with passed values
    for key, value in kwargs.items():
        if (key=='nPlottedTimesByYear'):
            nPlottedTimesByYear=value
        if (key=='xlabel'):
            xlabel=value
        if (key=='ylabel'):
            ylabel=value
        if (key=='zlabel'):
            zlabel=value
        if (key=='minyear'):
            minyear=value
        if (key=='maxyear'):
            maxyear=value
        if (key=='dateformat'):
            dateformat=value
        if (key=='axisFontSize'):
            axisFontSize=value
        if (key=='legendFontSize'):
            legendFontSize=value
        if (key=='ytick'):
            ytick=value

            
    min_date=datetime(minyear, 1, 1)
    max_date=datetime(maxyear, 1, 1)
    minTS=min_date.toordinal()
    maxTS=max_date.toordinal()
    
    # Ensure timeStamps are in datetime
    # If they are numeric, convert accordingly
    # For demonstration, assume they are datetime objects
    filtered_timeStamps = timeStamps[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    filtered_sigma = sigma[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    filtered_mu = mu[(timeStamps >= minTS) & (timeStamps <= maxTS)]
    

    fig = plt.figure()
    phandles = [fig]
    fig.set_size_inches(13, 7)

    L = len(filtered_timeStamps)
    
    # Compute number of points to plot
    avgYearLength = 365.2425
    total_years = (maxTS - minTS)/avgYearLength
    npdf = int(np.ceil(total_years * nPlottedTimesByYear))
    navg = int(np.ceil(L / npdf))
    
    plotSLength = npdf * navg
    timeStamps_plot = np.linspace(minTS, maxTS, plotSLength)

    # Handle epsilon
    if np.shape(epsilon) == ():  # scalar
        epsilon0 = np.ones(npdf) * epsilon
    else:
        epsilon_ = np.full(npdf * navg, np.nan)
        epsilon_flat = np.array(epsilon).flatten()
        epsilon_[:L] = epsilon_flat[:L]
        epsilonMtx = epsilon_.reshape(navg, -1, order='F') 
        epsilon0 = np.nanmean(epsilonMtx, axis=0)
        


    # Interpolate sigma and mu at plot points

    # Get interpolated values at desired points
    sigma_ = np.interp(timeStamps_plot, timeStamps, sigma)
    sigmaMtx = sigma_.reshape(navg, -1, order='F')
    sigma0 = np.nanmean(sigmaMtx, axis=0)
        
    mu_ = np.interp(timeStamps_plot, timeStamps, mu)
    muMtx = mu_.reshape(navg, -1, order='F')
    mu0 = np.nanmean(muMtx, axis=0)
    
    # Generate meshgrid for surface
    _, epsilonMtx = np.meshgrid(X, epsilon0)
    _, sigmaMtx = np.meshgrid(X, sigma0)
    XMtx, muMtx = np.meshgrid(X, mu0)

    timeStamps_plot = np.linspace(minTS, maxTS, len(mu0))
    # Compute GEV PDF
    gevvar = gev.pdf(XMtx, c=epsilonMtx, loc=muMtx, scale=sigmaMtx)

    # Plot surface
    ax = fig.add_subplot(111, projection='3d')
    X, timeStamps_plot = np.meshgrid(X, timeStamps_plot)
    surf = ax.plot_surface(X, timeStamps_plot, gevvar, cmap=cm.viridis, linewidth=0, antialiased=False)
             

    phandles.append(surf)

    # Date formatting on y-axis
    ax.yaxis.set_major_formatter(mdates.DateFormatter(dateformat))

    if ytick:
        ax.set_yticks(ytick)
        ax.set_yticklabels([datetime.fromordinal(int(t)).strftime(dateformat) for t in ytick])
    
    ax.set_ylim([minTS,maxTS])
    ax.view_init(elev=48, azim=24.3)

    ax.set_xlabel(xlabel, fontsize=labelFontSize, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=labelFontSize, labelpad=20)
    ax.set_zlabel(zlabel, fontsize=labelFontSize, labelpad=20)
    ax.tick_params(axis='both', labelsize=axisFontSize)

    plt.tight_layout()

    return phandles

def tsEvaNanRunningPercentile(series, windowSize, percent, **kwargs):
    series = np.array(series)
    length = len(series)
    
    # Configuration
    if windowSize > 2000: delta = 1.0
    elif windowSize > 1000: delta = 2.0
    elif windowSize > 100: delta = 5.0
    else: raise ValueError("window size cannot be less than 100")
        
    n_low_limit = kwargs.get('nLowLimit', 100)
    p_delta = kwargs.get('percentDelta', delta)
    pM_target, pP_target = percent - p_delta, percent + p_delta
    
    rn_prcnt0 = np.full(length, np.nan)
    dx = int(np.ceil(windowSize / 2))
    
    # 1. Initialize
    prob_obj = ProbObject(series[0 : dx + 1], pM_target, percent, pP_target)
    rn_prcnt0[0] = prob_obj.t
    
    # 2. Process
    for ii in range(1, length):
        min_idx = max(ii - dx, 0)
        max_idx = min(ii + dx, length - 1)
        
        # Remove exiting (left)
        if min_idx > 0:
            prob_obj.update(series[min_idx - 1], adding=False)

        # Add entering (right)
        if max_idx < length - 1:
            prob_obj.update(series[max_idx + 1], adding=True)

        # Re-initialize if the percentile drifted too far
        if prob_obj.is_out_of_bounds():
            prob_obj = ProbObject(series[min_idx : max_idx + 1], pM_target, percent, pP_target)
            
        # Interpolate
        if prob_obj.N > n_low_limit:
            rn_prcnt0[ii] = prob_obj.interpolate()

    # 3. Smooth and Return
    rnprcnt = tsEvaNanRunningMean(rn_prcnt0, windowSize)
    std_error = np.nanstd(rn_prcnt0 - rnprcnt)
    
    return rnprcnt, std_error
                                       
def tsEvaPlotSeriesTrendStdDevFromAnalysisObj(nonStationaryEvaParams,stationaryTransformData,**kwargs):
    plotPercentile = kwargs.get('plotPercentile',-1)
    ylabel = kwargs.get('ylabel','levels (m)')
    title = kwargs.get('title','')
    minYear = kwargs.get('minYear',1)
    maxYear = kwargs.get('maxYear',9999)
    for key, value in kwargs.items():
        if (key=='plotPercentile'): 
            plotPercentile=value
        if (key=='ylabel'): 
            ylabel=value
        if (key=='title'): 
            title=value
        if (key=='minYear'): 
            minYear=value
        if (key=='maxYear'): 
            maxYear=value

    # Extract required series
    timeStamps = stationaryTransformData.timeStamps
    series = stationaryTransformData.nonStatSeries
    trend = stationaryTransformData.trendSeries
    std_dev = stationaryTransformData.stdDevSeries
    
    if hasattr(stationaryTransformData, 'statsTimeStamps'):
        statsTimeStamps = stationaryTransformData.statsTimeStamps
    else:
        statsTimeStamps = timeStamps
    
    # Plot core trend + std dev
    phandles = tsEvaPlotSeriesTrendStdDev(timeStamps, series, trend, std_dev,statsTimeStamps=statsTimeStamps,**kwargs)

    # Optionally plot percentile
    if plotPercentile != -1:
        prcntile = tsEvaNanRunningPercentile(series,stationaryTransformData['runningStatsMulteplicity'],plotPercentile)

        fig = plt.figure(phandles[0].figure.number)
        ax = fig.gca()
        hndl, = ax.plot(timeStamps, prcntile)
        phandles.append(hndl)

    return phandles

def tsEvaPlotReturnLevelsGEV(epsilon, sigma, mu, epsilonStdErr, sigmaStdErr, muStdErr, **kwargs):
    # Default argument values

    minReturnPeriodYears = kwargs.get('minReturnPeriodYears',5)
    maxReturnPeriodYears = kwargs.get('maxReturnPeriodYears',1000)
    confidenceAreaColor = kwargs.get('confidenceAreaColor',[0.741, 0.988, 0.788])  # light green
    confidenceBarColor = kwargs.get('confidenceBarColor',[0.133, 0.545, 0.133])      # dark green
    returnLevelColor = kwargs.get('returnLevelColor','k')
    xlabel = kwargs.get('xlabel','return period (years)')
    ylabel = kwargs.get('ylabel','return levels (m)')
    ylim = kwargs.get('ylim',None)
    dtSampleYears = kwargs.get('dtSampleYears',1)
    ax = kwargs.get('ax',None)

    for key, value in kwargs.items():
        if (key=='minReturnPeriodYears'):
            minReturnPeriodYears=value
        if (key=='maxReturnPeriodYears'):
            maxReturnPeriodYears=value
        if (key=='confidenceAreaColor'):
            confidenceAreaColor=value
        if (key=='confidenceBarColor'):
            confidenceBarColor=value
        if (key=='returnLevelColor'):
            returnLevelColor=value
        if (key=='xlabel'):
            xlabel=value
        if (key=='ylabel'):
            ylabel=value
        if (key=='ylim'):
            ylim=value
        if (key=='dtSampleYears'):
            dtSampleYears=value
        if (key=='ax'):
            ax=value
    
    # Compute return periods and their corresponding periods in dt
    returnPeriodsInYears = np.logspace(np.log10(minReturnPeriodYears), np.log10(maxReturnPeriodYears), num=100)
    returnPeriodsInDts = returnPeriodsInYears / dtSampleYears

    # Compute return levels and errors
    returnLevels, returnLevelsErrs = tsEvaComputeReturnLevelsGEV(epsilon, sigma, mu, epsilonStdErr, sigmaStdErr, muStdErr, returnPeriodsInDts)
    
    # Confidence intervals
    supRLCI = returnLevels + 2 * returnLevelsErrs
    infRLCI = returnLevels - 2 * returnLevelsErrs
    
    # Determine Y limits based on the confidence intervals
    if ylim is not None:
        minRL = min(ylim)
        maxRL = max(ylim)
    else:
        minRL = min(np.min(arr) for arr in infRLCI)
        maxRL = max(np.min(arr) for arr in supRLCI)

        # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 7))  # Similar to figure position and size
    else:
        fig = None

    # Area plot for confidence intervals
    ax.fill_between(returnPeriodsInYears, infRLCI[0], supRLCI[0], color=confidenceAreaColor, label='Confidence Area', zorder=1)

    # Plot return levels
    ax.plot(returnPeriodsInYears, returnLevels[0], color=returnLevelColor, linewidth=3, label='Return Levels', zorder=2)

    # Plot confidence bars
    ax.plot(returnPeriodsInYears, supRLCI[0], color=confidenceBarColor, linewidth=2, label='Upper CI', zorder=3)
    ax.plot(returnPeriodsInYears, infRLCI[0], color=confidenceBarColor, linewidth=2, label='Lower CI', zorder=3)

    # Set plot scale to logarithmic for x-axis
    ax.set_xscale('log')

    # Set axis limits
    ax.set_xlim([minReturnPeriodYears, maxReturnPeriodYears])
#    ax.set_ylim([minRL[0], maxRL[0]])
    ax.set_ylim([minRL, maxRL])

    # Labeling and formatting
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.grid(True, which='both', zorder=4)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=16)


    # Finalizing plot appearance
    if fig:
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    phandles = {
        'fig': fig,
        'ax': ax,
        'confidence_area': ax.collections[0],  # first area collection
        'return_levels': ax.lines[0],  # first line plot (return levels)
        'upper_CI': ax.lines[1],  # second line plot (upper CI)
        'lower_CI': ax.lines[2],  # third line plot (lower CI)
    }

    return phandles

def tsEvaPlotReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, timeIndex, **kwargs):
    ylim = kwargs.get('ylim',None)
    for key, value in kwargs.items():
        if (key=='ylim'): 
            ylim=value

    epsilon = nonStationaryEvaParams[0]['parameters']['epsilon']
    sigma = nonStationaryEvaParams[0]['parameters']['sigma'][timeIndex] if isinstance(nonStationaryEvaParams[0]['parameters']['sigma'], np.ndarray) else nonStationaryEvaParams[0]['parameters']['sigma']
    mu = nonStationaryEvaParams[0]['parameters']['mu'][timeIndex] if isinstance(nonStationaryEvaParams[0]['parameters']['mu'], np.ndarray) else nonStationaryEvaParams[0]['parameters']['mu']
    dtSampleYears = nonStationaryEvaParams[0]['parameters']['timeDeltaYears']
    epsilonStdErr = nonStationaryEvaParams[0]['paramErr']['epsilonErr']
    sigmaStdErr = nonStationaryEvaParams[0]['paramErr']['sigmaErr'][timeIndex] if isinstance(nonStationaryEvaParams[0]['paramErr']['sigmaErr'], np.ndarray) else nonStationaryEvaParams[0]['paramErr']['sigmaErr']
    muStdErr = nonStationaryEvaParams[0]['paramErr']['muErr'][timeIndex] if isinstance(nonStationaryEvaParams[0]['paramErr']['muErr'], np.ndarray) else nonStationaryEvaParams[0]['paramErr']['muErr']
    
    phandles = tsEvaPlotReturnLevelsGEV(
        epsilon,
        sigma,
        mu,
        epsilonStdErr,
        sigmaStdErr,
        muStdErr,
        ylim=ylim
    )
    return phandles

                
def tsEvaPlotReturnLevelsGPDFromAnalysisObj(nonStationaryEvaParams, timeIndex, **kwargs):

    ylim = kwargs.get('ylim',None)
    for key, value in kwargs.items():
        if (key=='ylim'): 
            ylim=value

    epsilon = nonStationaryEvaParams[1]['parameters']['epsilon']
    sigma = nonStationaryEvaParams[1]['parameters']['sigma'][timeIndex] if isinstance(nonStationaryEvaParams[1]['parameters']['sigma'], np.ndarray) else nonStationaryEvaParams[1]['parameters']['sigma']
    threshold = nonStationaryEvaParams[1]['parameters']['threshold'][timeIndex] if isinstance(nonStationaryEvaParams[1]['parameters']['threshold'], np.ndarray) else nonStationaryEvaParams[1]['parameters']['threshold']
    thStart = nonStationaryEvaParams[1]['parameters']['timeHorizonStart']
    thEnd = nonStationaryEvaParams[1]['parameters']['timeHorizonEnd']
    timeHorizonInYears = round((thEnd-thStart)/ 365.2425)
    nPeaks = nonStationaryEvaParams[1]['parameters']['nPeaks']
    
    epsilonStdErr = nonStationaryEvaParams[1]['paramErr']['epsilonErr']
    sigmaStdErr = nonStationaryEvaParams[1]['paramErr']['sigmaErr'][timeIndex] if isinstance(nonStationaryEvaParams[1]['paramErr']['sigmaErr'], np.ndarray) else nonStationaryEvaParams[1]['paramErr']['sigmaErr']
    thresholdStdErr = nonStationaryEvaParams[1]['paramErr']['thresholdErr'][timeIndex] if isinstance(nonStationaryEvaParams[1]['paramErr']['thresholdErr'], np.ndarray) else nonStationaryEvaParams[1]['paramErr']['thresholdErr']

    phandles = tsEvaPlotReturnLevelsGPD(
        epsilon,
        sigma,
        threshold,
        epsilonStdErr,
        sigmaStdErr,
        thresholdStdErr,
        nPeaks,
        timeHorizonInYears,
        ylim=ylim
    )
    return phandles

def tsEvaPlotReturnLevelsGPD(epsilon, sigma, threshold, epsilonStdErr, sigmaStdErr,thresholdStdErr,nPeaks,timeHorizonInYears,**kwargs):
    # Default argument values

    minReturnPeriodYears = kwargs.get('minReturnPeriodYears',5)
    maxReturnPeriodYears = kwargs.get('maxReturnPeriodYears',1000)
    confidenceAreaColor = kwargs.get('confidenceAreaColor',[0.741, 0.988, 0.788])  # light green
    confidenceBarColor = kwargs.get('confidenceBarColor',[0.133, 0.545, 0.133])      # dark green
    returnLevelColor = kwargs.get('returnLevelColor','k')
    xlabel = kwargs.get('xlabel','return period (years)')
    ylabel = kwargs.get('ylabel','return levels (m)')
    ylim = kwargs.get('ylim',None)
    dtSampleYears = kwargs.get('dtSampleYears',1)
    ax = kwargs.get('ax',None)

    for key, value in kwargs.items():
        if (key=='minReturnPeriodYears'):
            minReturnPeriodYears=value
        if (key=='maxReturnPeriodYears'):
            maxReturnPeriodYears=value
        if (key=='confidenceAreaColor'):
            confidenceAreaColor=value
        if (key=='confidenceBarColor'):
            confidenceBarColor=value
        if (key=='returnLevelColor'):
            returnLevelColor=value
        if (key=='xlabel'):
            xlabel=value
        if (key=='ylabel'):
            ylabel=value
        if (key=='ylim'):
            ylim=value
        if (key=='dtSampleYears'):
            dtSampleYears=value
        if (key=='ax'):
            ax=value

    # Compute return periods and their corresponding periods in dt
    returnPeriodsInYears = np.logspace(np.log10(minReturnPeriodYears), np.log10(maxReturnPeriodYears), num=100)
    returnPeriodsInDts = returnPeriodsInYears / dtSampleYears

    # Compute return levels and errors (this should call your custom function)
    returnLevels, returnLevelsErrs = tsEvaComputeReturnLevelsGPD(epsilon, sigma, threshold, epsilonStdErr, sigmaStdErr, thresholdStdErr, nPeaks, timeHorizonInYears, returnPeriodsInDts)

    # Confidence intervals
    supRLCI = returnLevels + 2 * returnLevelsErrs
    infRLCI = returnLevels - 2 * returnLevelsErrs
    
    # Determine Y limits based on the confidence intervals
    if ylim is not None:
        minRL = min(ylim)
        maxRL = max(ylim)
    else:
        minRL = min(np.min(arr) for arr in infRLCI)
        maxRL = max(np.min(arr) for arr in supRLCI)

        
    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 7))  # Similar to figure position and size
    else:
        fig = None

    # Area plot for confidence intervals
    ax.fill_between(returnPeriodsInYears, infRLCI[0], supRLCI[0], color=confidenceAreaColor, label='Confidence Area', zorder=1)

    # Plot return levels
    ax.plot(returnPeriodsInYears, returnLevels[0], color=returnLevelColor, linewidth=3, label='Return Levels', zorder=2)

    # Plot confidence bars
    ax.plot(returnPeriodsInYears, supRLCI[0], color=confidenceBarColor, linewidth=2, label='Upper CI', zorder=3)
    ax.plot(returnPeriodsInYears, infRLCI[0], color=confidenceBarColor, linewidth=2, label='Lower CI', zorder=3)

    # Set plot scale to logarithmic for x-axis
    ax.set_xscale('log')

    # Set axis limits
    ax.set_xlim([minReturnPeriodYears, maxReturnPeriodYears])
#    ax.set_ylim([minRL[0], maxRL[0]])
    ax.set_ylim([minRL, maxRL])

    # Labeling and formatting
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.grid(True, which='both', zorder=4)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=16)


    # Finalizing plot appearance
    if fig:
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    phandles = {
        'fig': fig,
        'ax': ax,
        'confidence_area': ax.collections[0],  # first area collection
        'return_levels': ax.lines[0],  # first line plot (return levels)
        'upper_CI': ax.lines[1],  # second line plot (upper CI)
        'lower_CI': ax.lines[2],  # third line plot (lower CI)
    }

    return phandles

def tsEvaComputeTimeRP(params, RPiGEV, RPiGPD):
    paramx = pd.DataFrame(params, index=[0]).T
    qxV = 1 - np.exp(-((1 + paramx[0]['epsilonGEV'] * (RPiGEV - paramx[0]['muGEV']) / paramx[0]['sigmaGEV']) ** (-1 / paramx[0]['epsilonGEV'])))
    returnPeriodGEV = 1 / qxV
    X0 = paramx[0]['nPeaks'] / paramx[0]['SampleTimeHorizon']
    qxD = ((1 + paramx[0]['epsilonGPD'] * (RPiGPD - paramx[0]['thresholdGPD']) / paramx[0]['sigmaGPD']) ** (-1 / paramx[0]['epsilonGPD']))
    returnPeriodGPD = 1 / (X0 * qxD)

    return returnPeriodGEV, returnPeriodGPD

def tsEvaComputeReturnPeriodsGEV(epsilon, sigma, mu, BlockMax):
    nyr = len(BlockMax)
    # Compute the empirical return period:
    empval = empdis(BlockMax, nyr)
    my = [i for i, val in enumerate(empval['Q']) if val in BlockMax]
    PseudoObs = empval.iloc[my]

    # uniforming dimensions of yp, sigma, mu
    try:
        npars = len(sigma) if sigma is np.ndarray else 1
    except:
        raise Exception(f"tsEvaComputeReturnPeriodsGEV: unsupported type for sigma parameter: {type(sigma)}")
    nt = len(BlockMax)
    Bmax = np.tile(BlockMax, npars).reshape(npars, nt)
    sigma_ = np.empty([npars, nt])
    sigma_.fill(sigma)
    mu_ = np.empty([npars, nt])
    mu_.fill(mu)

    # estimating the return levels
    qx = 1 - np.exp(-(1 + epsilon * (Bmax - mu_) / sigma_)**(-1 / epsilon))
    GevPseudo = qx
    returnPeriods = 1 / qx
    return GevPseudo, returnPeriods, PseudoObs

def tsEvaComputeReturnPeriodsGPD(epsilon, sigma, threshold, peaks, nPeaks, peaksID, sampleTimeHorizon):
    X0 = nPeaks / sampleTimeHorizon
    nyr = sampleTimeHorizon

    peaksj = np.random.normal(peaks, 2)  # Jitter function replaced with numpy random normal
    peakAndId = pd.DataFrame({'peaksID': peaksID, 'peaksj': peaksj})
    peaksjid = sorted(peaksID, key=lambda x: peaksj.any())

    # Compute the empirical return period:
    empval = empdis(peaksj, nyr)  # Assuming empdis is a function defined elsewhere
    PseudoObs = empval
    PseudoObs['peaksID'] = peaksjid
    PseudoObs = PseudoObs.sort_values(by='peaksID')

    # Uniform dimensions of variables
    try:
        npars = len(sigma) if sigma is np.ndarray else 1
    except:
        raise Exception(f"tsEvaComputeReturnPeriodsGEV: unsupported type for sigma parameter: {type(sigma)}")    
    nt = len(peaks)
    Peakx = np.tile(PseudoObs['Q'],npars).reshape((npars, nt))
    sigma_ = np.empty([npars, nt])
    sigma_.fill(sigma)

    threshold_ = np.empty([npars, nt])
    threshold_.fill(threshold)

    # Estimate the return levels
    # Here I have all the pseudo observations of every annual extreme for every timestep
    qx = (((1 + epsilon * (Peakx - threshold_) / sigma_)**(-1 / epsilon)))
    GpdPseudo = qx
    GpdPseudo[GpdPseudo < 0] = 0
    returnPeriods = 1 / (X0 * qx)

    return GpdPseudo, returnPeriods, PseudoObs

def tsEvaComputeReturnLevelsGEV(epsilon,sigma,mu,epsilonStdErr,sigmaStdErr,muStdErr,returnPeriodsInDts):
    returnPeriodsInDts=np.matrix(returnPeriodsInDts)
    returnPeriodsInDtsSize = returnPeriodsInDts.shape
    if returnPeriodsInDtsSize[0] > 1:
        returnPeriodsInDts = returnPeriodsInDts.H
    yp = -np.log(1 - (1 / returnPeriodsInDts))
    npars = len(sigma) if sigma is np.ndarray else 1
    nt = len(returnPeriodsInDts) if returnPeriodsInDts is np.ndarray else 1
    yp = np.tile(yp, (npars, 1))
    yp=np.array(yp)
    sigma_ = np.tile(sigma, (nt, 1)).transpose()
    sigmaStdErr_ = np.tile(sigmaStdErr, (nt, 1)).transpose()

    mu_ = np.tile(mu, (nt, 1)).transpose()
    muStdErr_ = np.tile(muStdErr, (nt, 1)).transpose()
    if epsilon != 0:
        returnLevels = np.array(mu_ - ((sigma_ / epsilon) * (1 - (yp ** (-epsilon)))))
        dxm_mu = 1
        dxm_sigma = np.array((1/epsilon) * (1 - (yp ** (-epsilon))))
        dxm_epsilon = np.array(((sigma_ / (epsilon**2)) * (1 - (yp ** (-epsilon)))) - (((sigma_ / epsilon) * np.log(yp)) * (yp ** (-epsilon))))
        returnLevelsErr = np.array(((((dxm_mu * muStdErr_) ** 2) + ((dxm_sigma * sigmaStdErr_) ** 2))+ ((dxm_epsilon * epsilonStdErr) ** 2)) ** 0.5)
    else:
        returnLevels = np.array(mu_ - (sigma_ * np.log(yp)))
        dxm_u = 1
        dxm_sigma = np.array(np.log(yp))
        returnLevelsErr = np.array((((dxm_u * muStdErr_) ** 2) + ((dxm_sigma * sigmaStdErr_) ** 2)) ** 0.5)

    return returnLevels, returnLevelsErr

def tsEvaComputeReturnLevelsGEVFromAnalysisObj(nonStationaryEvaParams, returnPeriodsInYears, **kwargs):

    timeIndex = kwargs.get('timeIndex',-1)
    epsilon = nonStationaryEvaParams[0]['parameters']['epsilon']
    epsilonStdErr = nonStationaryEvaParams[0]['paramErr']['epsilonErr']
    epsilonStdErrFit = epsilonStdErr
    epsilonStdErrTransf = 0
    nonStationary = 'sigmaErrTransf' in nonStationaryEvaParams[0]['paramErr']
    
    if timeIndex > 0:
        sigma = nonStationaryEvaParams[0]['parameters']['sigma']
        mu = nonStationaryEvaParams[0]['parameters']['mu']
        sigmaStdErr = nonStationaryEvaParams[0]['paramErr']['sigmaErr']
        if 'sigmaErrFit' in nonStationaryEvaParams: sigmaStdErrFit = nonStationaryEvaParams[0]['paramErr']['sigmaErrFit']
        if 'sigmaErrTransf' in nonStationaryEvaParams: sigmaStdErrTransf = nonStationaryEvaParams[0]['paramErr']['sigmaErrTransf']
        muStdErr = nonStationaryEvaParams[0]['paramErr']['muErr']
        if 'muErrFit' in nonStationaryEvaParams: muStdErrFit = nonStationaryEvaParams[0]['paramErr']['muErrFit']
        if 'muErrTransf' in nonStationaryEvaParams: muStdErrTransf = nonStationaryEvaParams[0]['paramErr']['muErrTransf']
    else:
        sigma = nonStationaryEvaParams[0]['parameters']['sigma']
        mu = nonStationaryEvaParams[0]['parameters']['mu']
        sigmaStdErr = nonStationaryEvaParams[0]['paramErr']['sigmaErr']
        muStdErr = nonStationaryEvaParams[0]['paramErr']['muErr']
        
        if nonStationary:
            if 'sigmaErrFit' in nonStationaryEvaParams: sigmaStdErrFit = nonStationaryEvaParams[0]['paramErr']['sigmaErrFit']
            if 'sigmaErrTransf' in nonStationaryEvaParams: sigmaStdErrTransf = nonStationaryEvaParams[0]['paramErr']['sigmaErrTransf']
            if 'muErrFit' in nonStationaryEvaParams: muStdErrFit = nonStationaryEvaParams[0]['paramErr']['muErrFit']
            if 'muErrTransf' in nonStationaryEvaParams: muStdErrTransf = nonStationaryEvaParams[0]['paramErr']['muErrTransf']

    returnLevels, returnLevelsErr = tsEvaComputeReturnLevelsGEV(epsilon, sigma, mu, epsilonStdErr, sigmaStdErr, muStdErr, returnPeriodsInYears)
    
    
#    if nonStationary:
#        returnLevels, returnLevelsErrFit = tsEvaComputeReturnLevelsGEV(epsilon, sigma, mu, epsilonStdErrFit, sigmaStdErrFit, muStdErrFit, returnPeriodsInYears)
#        returnLevels, returnLevelsErrTransf = tsEvaComputeReturnLevelsGEV(epsilon, sigma, mu, epsilonStdErrTransf, sigmaStdErrTransf, muStdErrTransf, returnPeriodsInYears)
#    else:
#        returnLevelsErrFit = returnLevelsErr
#        returnLevelsErrTransf = [0] * returnLevelsErr.size
    return returnLevels, returnLevelsErr


def tsEvaComputeReturnLevelsGPD(epsilon, sigma, threshold, epsilonStdErr, sigmaStdErr, thresholdStdErr, nPeaks, sampleTimeHorizon, returnPeriods):
    X0 = nPeaks / sampleTimeHorizon
    XX = X0 * np.array(returnPeriods)
    if isinstance(sigma, np.ndarray):
        npars = len(sigma)
    else:
        npars = 1

    nt = len(returnPeriods)
    XX_ = np.tile(XX, (npars, 1))
    XXlog_=np.log(XX_)
    sigma_ = np.tile(sigma, (nt, 1)).transpose()
    sigmaStdErr_ = np.tile(sigmaStdErr, (nt, 1)).transpose()
    threshold_ = np.tile(threshold, (nt, 1)).transpose()
    thresholdStdErr_ = np.tile(thresholdStdErr, (nt, 1)).transpose()
    
    if epsilon != 0:
        # estimating the return levels
        returnLevels = threshold_ + sigma_ / epsilon * ((XX_)**epsilon - 1)
        # estimating the error
        # estimating the differential of returnLevels to the parameters
        # !! ASSUMING NON ZERO ERROR ON THE THRESHOLD AND 0 ERROR ON THE PERCENTILE.
        # THIS IS NOT COMPLETELY CORRECT BECAUSE THE PERCENTILE DEPENDS ON THE
        # THRESHOLD AND HAS THEREFORE AN ERROR RELATED TO THAT OF THE
        # THRESHOLD
        dxm_u = 1
        dxm_sigma = 1 / epsilon * ((XX_)**epsilon - 1)
        dxm_epsilon = -sigma_ / epsilon**2 * ((XX_)**epsilon - 1) + sigma_ / epsilon * np.log(XX_) * (XX_)**epsilon
        
        returnLevelsErr = ((dxm_u * thresholdStdErr_)**2 + (dxm_sigma * sigmaStdErr_)**2 + (dxm_epsilon * epsilonStdErr)**2)**0.5
    else:
        returnLevels = threshold_ + sigma_ * np.log(XX_)
        dxm_u = 1
        dxm_sigma = np.log(XX_)
        
        returnLevelsErr = ((dxm_u * thresholdStdErr_)**2 + (dxm_sigma * sigmaStdErr_)**2)**0.5
        
    return returnLevels, returnLevelsErr

def tsEvaComputeReturnLevelsGPDFromAnalysisObj(nonStationaryEvaParams, returnPeriodsInYears, **kwargs):

    timeIndex = kwargs.get('timeIndex',-1)
    epsilon = nonStationaryEvaParams[1]['parameters']['epsilon']
    epsilonStdErr = nonStationaryEvaParams[1]['paramErr']['epsilonErr']
    epsilonStdErrFit = epsilonStdErr
    epsilonStdErrTransf = 0
    thStart = nonStationaryEvaParams[1]['parameters']['timeHorizonStart']
    thEnd = nonStationaryEvaParams[1]['parameters']['timeHorizonEnd']
    timeHorizonInYears = round((thEnd-thStart)/ 365.2425)
    nPeaks = nonStationaryEvaParams[1]['parameters']['nPeaks']
    nonStationary = "sigmaErrTransf" in nonStationaryEvaParams[1]['paramErr']
    if timeIndex > 0:
        sigma = nonStationaryEvaParams[1]['parameters']['sigma']
        threshold = nonStationaryEvaParams[1]['parameters']['threshold']
        sigmaStdErr = nonStationaryEvaParams[1]['paramErr']['sigmaErr']
        if 'sigmaErrFit' in nonStationaryEvaParams: sigmaStdErrFit = nonStationaryEvaParams[1]['paramErr']['sigmaErrFit']
        if 'sigmaErrTransf' in nonStationaryEvaParams: sigmaStdErrTransf = nonStationaryEvaParams[1]['paramErr']['sigmaErrTransf']
        thresholdStdErr = nonStationaryEvaParams[1]['paramErr']['thresholdErr']
        thresholdStdErrFit = 0
        if 'thresholdErrTransf' in nonStationaryEvaParams: thresholdStdErrTransf = nonStationaryEvaParams[1]['paramErr']['thresholdErrTransf']
    else:
        sigma = nonStationaryEvaParams[1]['parameters']['sigma']
        threshold = nonStationaryEvaParams[1]['parameters']['threshold']
        sigmaStdErr = nonStationaryEvaParams[1]['paramErr']['sigmaErr']
        if nonStationary:
            thresholdStdErr = nonStationaryEvaParams[1]['paramErr']['thresholdErr']
            
            if 'sigmaErrFit' in nonStationaryEvaParams: sigmaStdErrFit = nonStationaryEvaParams[1]['paramErr']['sigmaErrFit']
            if 'sigmaErrTransf' in nonStationaryEvaParams: sigmaStdErrTransf = nonStationaryEvaParams[1]['paramErr']['sigmaErrTransf']
            if 'thresholdErrFit' in nonStationaryEvaParams: thresholdStdErrFit = nonStationaryEvaParams[1]['paramErr']['thresholdErrFit']
            if 'thresholdErrTransf' in nonStationaryEvaParams: thresholdStdErrTransf = nonStationaryEvaParams[1]['paramErr']['thresholdErrTransf']
        else:
            thresholdStdErr = 0
            
    returnLevels,returnLevelsErr = tsEvaComputeReturnLevelsGPD(epsilon, sigma, threshold, epsilonStdErr, sigmaStdErr, thresholdStdErr, nPeaks, timeHorizonInYears, returnPeriodsInYears)

#    if nonStationary:
#        returnLevelsErrFit = tsEvaComputeReturnLevelsGPD(epsilon, sigma, threshold, epsilonStdErrFit, sigmaStdErrFit, [0]*len(thresholdStdErr), nPeaks, timeHorizonInYears, returnPeriodsInYears)
#        returnLevelsErrTransf = tsEvaComputeReturnLevelsGPD(epsilon, sigma, threshold, epsilonStdErrTransf, sigmaStdErrTransf, thresholdStdErrTransf, nPeaks, timeHorizonInYears, returnPeriodsInYears)
#    else:
#        returnLevelsErrFit = returnLevelsErr
#        returnLevelsErrTransf = [0]*returnLevelsErr.size
 
    return returnLevels, returnLevelsErr#, returnLevelsErrFit,returnLevelsErrTransf

def tsEvaComputeRLsGEVGPD(nonStationaryEvaParams, RPgoal, timeIndex, trans=None):

    # GEV
    epsilonGEV = -nonStationaryEvaParams['GetStat']['parameters']['epsilonGEV']
    sigmaGEV = np.mean(nonStationaryEvaParams['GetStat']['parameters']['sigmaGEV'][timeIndex])
    muGEV = np.mean(nonStationaryEvaParams['GetStat']['parameters']['muGEV'][timeIndex])
    dtSampleYears = nonStationaryEvaParams['GetStat']['parameters']['timeDeltaYears']

    # GPD
    epsilonGPD = -nonStationaryEvaParams[1]['parameters']['epsilonGPD']
    sigmaGPD = np.mean(nonStationaryEvaParams[1]['parameters']['sigmaGPD'][timeIndex])
    thresholdGPD = np.mean(nonStationaryEvaParams[1]['parameters']['thresholdGPD'][timeIndex])
    nPeaks = nonStationaryEvaParams[1]['parameters']['nPeaks']
    thStart = nonStationaryEvaParams[1]['parameters']['timeHorizonStart']
    thEnd = nonStationaryEvaParams[1]['parameters']['timeHorizonEnd']
    sampleTimeHorizon = round((thEnd - thStart).dt.days / 365.2425)

    if nonStationaryEvaParams['GetStat']['method'] == "No fit":
        print("Could not fit EVD to this pixel")
        ParamGEV = np.array([epsilonGEV, sigmaGEV, muGEV, None, None, None])
        ParamGPD = np.array([epsilonGPD, sigmaGPD, thresholdGPD, None, None, None, nPeaks, sampleTimeHorizon])
        return {'Fit': 'No fit', 'Params': [ParamGEV, ParamGPD]}
    else:
        epsilonStdErrGEV = nonStationaryEvaParams['GetStat']['paramErr']['epsilonGEVErr']
        sigmaStdErrGEV = np.mean(nonStationaryEvaParams['GetStat']['paramErr']['sigmaGEVErr'][timeIndex])
        muStdErrGEV = np.mean(nonStationaryEvaParams['GetStat']['paramErr']['muGEVErr'][timeIndex])

        epsilonStdErrGPD = nonStationaryEvaParams[1]['paramErr']['epsilonGPDErr']
        sigmaStdErrGPD = np.mean(nonStationaryEvaParams[1]['paramErr']['sigmaGPDErr'][timeIndex])
        thresholdStdErrGPD = np.mean(nonStationaryEvaParams[1]['paramErr']['thresholdGPDErr'][timeIndex])

        if trans == "rev":
            sigmaGEV = -sigmaGEV
            muGEV = -muGEV
            sigmaGPD = -sigmaGPD
            thresholdGPD = -thresholdGPD

        returnLevelsGEV,returnLevelsGEVErr = tsEvaComputeReturnLevelsGEV(epsilonGEV, sigmaGEV, muGEV, epsilonStdErrGEV, sigmaStdErrGEV, muStdErrGEV, RPgoal)
        returnLevelsGPD,returnLevelsGPDErr = tsEvaComputeReturnLevelsGPD(epsilonGPD, sigmaGPD, thresholdGPD, epsilonStdErrGPD, sigmaStdErrGPD, thresholdStdErrGPD, nPeaks, sampleTimeHorizon, RPgoal)

        rlevGEV = returnLevelsGEV
        rlevGPD = returnLevelsGPD
        errGEV = returnLevelsGEVErr
        errGPD = returnLevelsGPDErr

        ParamGEV = [epsilonGEV, sigmaGEV, muGEV, epsilonStdErrGEV, sigmaStdErrGEV, muStdErrGEV]
        ParamGPD = [epsilonGPD, sigmaGPD, thresholdGPD, epsilonStdErrGPD, sigmaStdErrGPD, thresholdStdErrGPD, nPeaks, sampleTimeHorizon]

        return 'Fitted',RPgoal,rlevGEV,rlevGPD,errGEV,errGPD, ParamGEV, ParamGPD

def tsTimeSeriesToPointData(ms, pot_threshold, pot_threshold_error):
    # Filter the time series by the threshold
    ms1 = ms[ms[:, 1] > pot_threshold]
    percentile = (1 - ms1.shape[0] / ms.shape[0]) * 100

    pointData = {}
    pointData['completeSeries'] = ms1

    pot_data = {
        'threshold': pot_threshold,
        'thresholdError': pot_threshold_error,
        'percentile': percentile,
        'peaks': ms1[:, 1],
        'ipeaks': np.arange(1, ms1.shape[0] + 1),
        'sdpeaks': ms1[:, 0]
    }
    pointData['POT'] = pot_data

    # These need to be implemented or imported
    ret_annual = tsEvaComputeAnnualMaxima(ms1)

    pointData['annualMax']=ret_annual['annualMax']
    pointData['annualMaxTimeStamp']=ret_annual['annualMaxDate']
    pointData['annualMaxIndexes']=ret_annual['annualMaxIndx']

    ret_monthly = tsEvaComputeMonthlyMaxima(ms1)

    pointData['monthlyMax']=ret_monthly['monthlyMax']
    pointData['monthlyMaxTimeStamp']=ret_monthly['monthlyMaxDate']
    pointData['monthlyMaxIndx']=ret_monthly['monthlyMaxIndx']
    
    # Compute the years range
    datetime_series_with_offset = pd.to_datetime(np.array(ms1[:, 0]) - 719529, unit='D', origin='unix')  + pd.Timedelta(hours=1)
    yrs = np.unique(datetime_series_with_offset.year)
    yrs = yrs - np.min(yrs)
    pointData['years'] = np.arange(np.nanmin(yrs), np.nanmax(yrs) + 1)

    pointData['Percentiles'] = [0]

    return pointData

def tsEvaSampleData(ms, **kwargs):
    pctsDesired = [90, 95, 99, 99.9]
    meanEventsPerYear=kwargs.get('meanEventsPerYear',5)
    potPercentiles=kwargs.get('potPercentiles',[50, 70] + list(range(85, 98, 2)))

    for key, value in kwargs.items():
        if (key=='meanEventsPerYear'): 
            meanEventsPerYear=value
        if (key=='potPercentiles'): 
            potPercentiles=value


    POTData = tsGetPOT(ms, potPercentiles, meanEventsPerYear, **kwargs)

    vals = np.nanquantile(ms[:, 1], [x / 100 for x in pctsDesired])
    
    percentiles = {'percentiles': pctsDesired, 'values': vals}

    pointData = {}
    pointData['completeSeries'] = ms
    pointData['POT'] = POTData
    pointDataA = tsEvaComputeAnnualMaxima(ms)
    pointDataM = tsEvaComputeMonthlyMaxima(ms)

    datetime_series_with_offset = pd.to_datetime(np.array(ms[:, 0]) - 719529, unit='D', origin='unix')  + pd.Timedelta(hours=1)
    yrs = np.unique(datetime_series_with_offset.year)
    yrs=yrs-np.min(yrs)
    pointData['years'] = list(range(min(yrs), max(yrs) + 1))
    pointData['Percentiles'] = percentiles
    pointData['annualMax'] = pointDataA['annualMax']
    pointData['annualMaxDate'] = pointDataA['annualMaxDate']
    pointData['annualMaxIndx'] = pointDataA['annualMaxIndx']
    pointData['monthlyMax'] = pointDataM['monthlyMax']
    pointData['monthlyMaxDate'] = pointDataM['monthlyMaxDate']
    pointData['monthlyMaxIndx'] = pointDataM['monthlyMaxIndx']

    return pointData

def tsGetPOT(ms, pcts, desiredEventsPerYear, **kwargs):
    
    minPeakDistanceInDays=kwargs.get('minPeakDistanceInDays',3)
    for key, value in kwargs.items():
        if (key=='minPeakDistanceInDays'): 
            minPeakDistanceInDays=value
    if minPeakDistanceInDays == -1:
        raise ValueError("label parameter 'minPeakDistanceInDays' must be set")

    ms[np.isnan(ms[:, 1]), 1] = 0
    dt1 = tsEvaGetTimeStep(ms[:, 0])
    dt = dt1.astype(float)
    
    minPeakDistance = minPeakDistanceInDays / dt
     
    nyears = round((np.max(ms[:, 0]) - np.min(ms[:, 0])) / 365.25)
        
    if len(pcts) == 1:
        pcts = [pcts - 3, pcts]
        desiredEventsPerYear = -1
    
    numperyear = np.empty(len(pcts))
    minnumperyear = np.empty(len(pcts))
    thrsdts = np.empty(len(pcts))
    gpp = np.empty(len(pcts))
    gpp[:]=np.nan
    devpp = np.empty(len(pcts))
    devpp[:]=np.nan
    trip = None
    perfpen = 0

    for ipp in range(len(pcts)):
        thrsdt = np.percentile(ms[:, 1], pcts[ipp])
        thrsdts[ipp] = thrsdt
        ms[np.isnan(ms[:, 1])] = -9999
        minEventsPerYear = 1
        
        #        if tail == "high":
        shape_bnd = [-0.5, 1]
        locs,pks = find_peaks(ms[:, 1], height=thrsdt,distance=minPeakDistance)
            
#        if tail == "low":
#            shape_bnd = [-1.5, 0]
#            locs,pks = declustpeaks(data = ms[:, 1] ,minpeakdistance = minPeakDistance ,minrundistance = minRunDistance, qt=thrsdt)

        peaks=pks['peak_heights']

        numperyear[ipp] = len(peaks) / nyears
        
        nperYear=tsGetNumberPerYear(ms,locs);
        minnumperyear[ipp]=np.nanmin(nperYear);

        if ipp>0 and numperyear[ipp] < minEventsPerYear and minnumperyear[ipp]<minEventsPerYear:
            break
        
    diffNPerYear = np.mean(np.diff(np.nan_to_num(numperyear[::-1])))
    
    if diffNPerYear == 0:
        diffNPerYear = 1
    
    thresholdError = np.mean(np.diff(np.nan_to_num(thrsdts))) / diffNPerYear / 2
    indexp = ipp

    if indexp is not None:
        thrsd = np.quantile(ms[:, 1], pcts[indexp] / 100)
        pct = pcts[indexp]
    else:
        thrsd = 0
        pct = None
        
    #    if tail == "high":
    locs,pks = find_peaks(ms[:, 1], distance=minPeakDistance, height=thrsdt)
#    if tail == "low":
#        locs,pks = declustpeaks(data=ms[:, 1], minpeakdistance=minPeakDistance, minrundistance=minRunDistance, qt=thrsd)
    peaks=pks['peak_heights']
    
    ms_Q=ms[:, 1]
    ms_time=ms[:, 0]+3600
    peaks = np.array(ms_Q[locs])
    st = locs[0]
    end = locs[len(locs)-1]

    POTdata = {
        'threshold': thrsd,
        'thresholdError': thresholdError,
        'percentile': pct,
        'peaks': peaks,
        'stpeaks': st,
        'endpeaks': end,
        'ipeaks': locs,
        'time': np.array(ms_time[locs]),
    }

    return POTdata

def tsEvaGetTimeStep(times):
    df = np.diff(times)  # Compute the differences between consecutive times
    dt = np.min(df)      # Find the minimum difference
    if dt == 0:
        df = df[df != 0]  # Remove zeros from the differences
        dt = np.min(df)   # Find the minimum difference again
    return dt

import numpy as np

def tsSameValuesSegmentation(iii, val=1):
    # Initialize the output lists
    inds = []
    rinds = []

    # Length of the input array
    ll = len(iii)

    # Calculate the differences between consecutive elements
    diffs = np.diff(iii)
    
    # Find the indices where the value changes
    indsep = np.where(diffs != 0)[0]

    # Define the start (l1) and end (l2) indices for each segment
    l1 = np.concatenate(([0], indsep + 1))
    l2 = np.concatenate((indsep, [ll - 1]))

    # Iterate through the segments
    for i in range(len(l1)):
        if iii[l1[i]] == val:
            inds.append(iii[l1[i]:l2[i] + 1])  # Extract the segment of values
            rinds.append(np.arange(l1[i], l2[i] + 1))  # Extract the indices of the segment

    return inds, rinds

def tsEvaFillSeries(timeStamps, series):
    mask = ~np.isnan(series)
    ts_clean = np.array(timeStamps)[mask]
    s_clean = np.array(series)[mask]

    df = pd.DataFrame({'ts': ts_clean, 'val': s_clean}).sort_values('ts')
    df_unique = df.groupby('ts')['val'].max().reset_index()
    
    new_ts = df_unique['ts'].values
    new_series = df_unique['val'].values
    
    min_t = np.min(new_ts)
    max_t = np.max(new_ts)
    diffs = np.diff(new_ts)
    dt = np.min(diffs) if len(diffs) > 0 else 0

    if 350 <= dt <= 370:
        # Annual series
        start_year = pd.to_datetime(min_t, unit='D', origin='719529').year
        end_year = pd.to_datetime(max_t, unit='D', origin='719529').year
        filled_dt = pd.date_range(start=f"{start_year}-01-01", 
                                  end=f"{end_year}-01-01", freq='YS')
        filled_time_stamps = (filled_dt - pd.Timestamp("0000-01-01")).days + 366
        
    elif 28 <= dt <= 31:
        # Monthly series
        start_date = pd.to_datetime(min_t, unit='D', origin='719529')
        end_date = pd.to_datetime(max_t, unit='D', origin='719529')
        filled_dt = pd.date_range(start=f"{start_date.year}-{start_date.month}-01", 
                                  end=f"{end_date.year}-{end_date.month}-01", freq='MS')
        filled_time_stamps = (filled_dt - pd.Timestamp("0000-01-01")).days + 366
        
    else:
        # Linear spacing
        filled_time_stamps = np.arange(min_t, max_t + dt, dt)

    f = interp1d(new_ts, new_series, kind='nearest', fill_value="extrapolate")
    filled_series = f(filled_time_stamps)

    filled_series = tsRemoveConstantSubseries(filled_series, 4)

    return filled_time_stamps, filled_series, dt

def tsRemoveConstantSubseries(srs, stackedValuesCount):
    series = np.array(srs, dtype=float)
    count = 1
    for i in range(1, len(series)):
        if series[i] == series[i-1]:
            count += 1
        else:
            if count >= stackedValuesCount:
                series[i-count:i] = np.nan
            count = 1
    # Check last group
    if count >= stackedValuesCount:
        series[-count:] = np.nan
    return series

def tsEvaNanRunningMean(series, windowSize):
    minNThreshold = 1
    rnmn = np.full(len(series), np.nan)  # Initialize output with NaNs
    dx = int(np.ceil(windowSize / 2))  # Half window size, rounded up
    l = len(series)
    sm = 0
    n = 0

    for ii in range(l):
        minidx = max(ii - dx, 0)  # Adjust for 0-based index
        maxidx = min(ii + dx, l - 1)  # Adjust for 0-based index
        
        if ii == 0:  # For the first element
            subsrs = series[minidx:maxidx + 1]  # Slicing inclusive
            sm = np.nansum(subsrs)
            n = np.sum(~np.isnan(subsrs))
        else:
            if minidx > 0:  # If we're not at the start
                sprev = series[minidx - 1]
                if not np.isnan(sprev):
                    sm -= sprev
                    n -= 1
            if maxidx < l - 1:  # If we're not at the end
                snext = series[maxidx + 1]
                if not np.isnan(snext):
                    sm += snext
                    n += 1
        
        if n > minNThreshold:
            rnmn[ii] = sm / n
        else:
            rnmn[ii] = np.nan

    return rnmn

def tsEvaRunningMeanTrend(timeStamps, series, timeWindow):
    filledTimeStamps, filledSeries, dt = tsEvaFillSeries(timeStamps, series)
    nRunMn = np.ceil(timeWindow/dt)
    trendSeries = tsEvaNanRunningMean(filledSeries, nRunMn)
    trendSeries = tsEvaNanRunningMean(trendSeries, np.ceil(nRunMn / 2))
    return trendSeries, filledTimeStamps, filledSeries, nRunMn

def tsEvaDetrendTimeSeries(timeStamps, series, timeWindow, **kwargs):
    extremeLowThreshold=kwargs.get('extremeLowThreshold',float('-inf'))
    for key, value in kwargs.items():
        if (key=='extremeLowThreshold'): 
            extremeLowThreshold=value
    trendSeries, filledTimeStamps, filledSeries, nRunMn = tsEvaRunningMeanTrend(
        timeStamps, series, timeWindow
    )
    statSeries = filledSeries.copy()
    statSeries[statSeries < extremeLowThreshold] = np.nan
    detrendSeries = statSeries - trendSeries
    return detrendSeries, trendSeries, filledTimeStamps, filledSeries, nRunMn

def tsEvaNanRunningStatistics(series, windowSize):
    minNThreshold = 1

    # Calculate running mean using the separate function
    rnmn = tsEvaNanRunningMean(series, windowSize)
    
    # Initialize output arrays with NaNs
    rnvar = np.full(len(series), np.nan)
    rn3mom = np.full(len(series), np.nan)
    rn4mom = np.full(len(series), np.nan)

    dx = int(np.ceil(windowSize / 2))
    l = len(series)

    sm = 0
    smsq = 0 
    sm3pw = 0
    sm4pw = 0
    n = 0    

    for ii in range(l): # Python's 0-based indexing
        minindx = max(ii - dx, 0)
        maxindx = min(ii + dx, l - 1)

        if ii == 0:
            subsrs = series[minindx : maxindx + 1]

            subsrsMean = rnmn[ii] 
            
            if not np.isnan(subsrsMean):
                diffFromMean = subsrs - subsrsMean
                
                validDiffs = diffFromMean[~np.isnan(diffFromMean)]
                smsq = np.sum(validDiffs**2)
                sm3pw = np.sum(validDiffs**3)
                sm4pw = np.sum(validDiffs**4)
                n = len(validDiffs)
            else:
                smsq, sm3pw, sm4pw, n = 0, 0, 0, 0
        else:

            if minindx > 0:
                sprevVal = series[minindx - 1]
                sprevMean = rnmn[minindx - 1]
                
                if not np.isnan(sprevVal) and not np.isnan(sprevMean):
                    sprevDiff = sprevVal - sprevMean
                    smsq = max(0, smsq - sprevDiff**2)
                    sm3pw = sm3pw - sprevDiff**3
                    sm4pw = max(0, sm4pw - sprevDiff**4)
                    n -= 1

            # Element entering the window from the right
            if maxindx < l - 1:
                snextVal = series[maxindx + 1]
                snextMean = rnmn[minindx + 1]
                
                if not np.isnan(snextVal) and not np.isnan(snextMean):
                    snextDiff = snextVal - snextMean
                    smsq += snextDiff**2
                    sm3pw += snextDiff**3
                    sm4pw += snextDiff**4
                    n += 1
        
        # Calculate moments if enough non-NaN elements are present
        
        if n > minNThreshold:
            rnvar[ii] = smsq / n
            rn3mom[ii] = sm3pw / n
            rn4mom[ii] = sm4pw / n
        else:
            pass

    return rnmn, rnvar, rn3mom, rn4mom

def tsEvaNanRunningVariance(series, windowSize):

    minNThreshold = 1

    rnmn = np.full(len(series), np.nan) # Initialize with NaNs
    dx = int(np.ceil(windowSize / 2))
    l = len(series)
    smsq = 0
    n = 0

    for ii in range(l):
        minindx = max(ii - dx, 0) # Python uses 0-based indexing
        maxindx = min(ii + dx, l - 1) # Python's slice end is exclusive, but for direct indexing it's l-1

        if ii == 0: 
            subSqSrs = series[minindx : maxindx + 1]**2
            smsq = np.nansum(subSqSrs)
            n = np.sum(~np.isnan(subSqSrs))
        else:
            # When window slides, subtract the squared value leaving the window
            if minindx > 0: # If the left side of the window moved
                sprev = series[minindx - 1]
                if not np.isnan(sprev):
                    smsq = max(0, smsq - sprev**2)
                    n -= 1
            
            # Add the squared value entering the window
            if maxindx < l - 1: # If the right side of the window can expand
                snext = series[maxindx + 1]
                if not np.isnan(snext):
                    smsq += snext**2
                    n += 1
        
        if n > minNThreshold:
            rnmn[ii] = smsq / n
        else:
            rnmn[ii] = np.nan 
            
    return rnmn

def tsEvaTransformSeriesToStationaryTrendOnly(timeStamps, series, timeWindow, **kwargs):
    print("computing the trend ...")
    class TrasfData:
        pass
    
    trasfData = TrasfData()

    (
        statSeries,
        trendSeries,
        filledTimeStamps,
        filledSeries,
        nRunMn,
    ) = tsEvaDetrendTimeSeries(timeStamps, series, timeWindow)
    print("computing the slowly varying standard deviation ...")
    varianceSeries = tsEvaNanRunningVariance(statSeries, nRunMn)
    varianceSeries = tsEvaNanRunningMean(varianceSeries, np.ceil(nRunMn / 2))
    stdDevSeries = varianceSeries**0.5
    statSeries = statSeries / stdDevSeries
    _, _, statSer3Mom, statSer4Mom = tsEvaNanRunningStatistics(statSeries, nRunMn)
    statSer3Mom = tsEvaNanRunningMean(statSer3Mom, np.ceil(nRunMn))
    statSer4Mom = tsEvaNanRunningMean(statSer4Mom, np.ceil(nRunMn))

    N = nRunMn.copy()
    trendError = np.nanmean(stdDevSeries) / (N ** 0.5)
    avgStdDev = np.nanmean(stdDevSeries)
    S = 2
    stdDevError = avgStdDev * (2 * S**2 / N**3) ** (1.0 / 4.0)
    
    
    trasfData.runningStatsMulteplicity = nRunMn.copy()
    trasfData.stationarySeries = statSeries.copy()
    trasfData.trendSeries = trendSeries.copy()
    trasfData.trendSeriesNonSeasonal = trendSeries.copy()
    trasfData.trendError = trendError.copy()
    trasfData.stdDevSeries = stdDevSeries.copy()
    trasfData.stdDevSeriesNonSeasonal = stdDevSeries.copy()
    trasfData.stdDevError = stdDevError * np.ones_like(stdDevSeries)
    trasfData.timeStamps = filledTimeStamps.copy()
    trasfData.nonStatSeries = filledSeries.copy()
    trasfData.statSer3Mom = statSer3Mom.copy()
    trasfData.statSer4Mom = statSer4Mom.copy()

    return trasfData

def tsEstimateAverageSeasonality(time_stamps, seasonality_series):
    avg_year_length = 365.2425
    n_month_in_year = 12
    avg_month_length = avg_year_length / n_month_in_year

    first_tm_stamp = time_stamps[0]
    last_tm_stamp = time_stamps[-1]
    month_tm_stamp_start = np.arange(first_tm_stamp, last_tm_stamp, avg_month_length)
    month_tm_stamp_end = month_tm_stamp_start + avg_month_length

    grpd_ssn_ = [np.nanmean(seasonality_series[(month_tm_stamp_start[i] <= time_stamps) & (time_stamps < month_tm_stamp_end[i])]) for i in range(len(month_tm_stamp_start))]
    n_years = int(np.ceil(len(grpd_ssn_) / n_month_in_year))
    grpd_ssn = np.full(n_years * n_month_in_year, np.nan)
    grpd_ssn[:len(grpd_ssn_)] = grpd_ssn_

    grpd_ssn_mtx = grpd_ssn.reshape(n_month_in_year, -1, order='F')
    mn_ssn_ = np.nanmean(grpd_ssn_mtx, axis=1)

    # estimating the first 2 Fourier components
    it = np.arange(1, n_month_in_year + 1)
    x = it / 6 * np.pi
    dx = np.pi / 6
    a0 = np.mean(mn_ssn_)
    a1 = (1 / np.pi) * np.sum(np.cos(x) * mn_ssn_) * dx
    b1 = (1 / np.pi) * np.sum(np.sin(x) * mn_ssn_) * dx
    a2 = (1 / np.pi) * np.sum(np.cos(2 * x) * mn_ssn_) * dx
    b2 = (1 / np.pi) * np.sum(np.sin(2 * x) * mn_ssn_) * dx
    a3 = (1 / np.pi) * np.sum(np.cos(3 * x) * mn_ssn_) * dx
    b3 = (1 / np.pi) * np.sum(np.sin(3 * x) * mn_ssn_) * dx

    mn_ssn = a0 + (a1 * np.cos(x) + b1 * np.sin(x)) + (a2 * np.cos(2 * x) + b2 * np.sin(2 * x)) + (a3 * np.cos(3 * x) + b3 * np.sin(3 * x))
    month_avg_mtx = np.tile(mn_ssn[:, np.newaxis], (1, n_years))
    month_avg_vec = month_avg_mtx.flatten(order='F')

    imnth = np.arange(len(month_avg_vec))
    avg_tm_stamp = first_tm_stamp + avg_month_length / 2 + imnth * avg_month_length

    # adding first and last times
    month_avg_vec = np.concatenate(([month_avg_vec[0]], month_avg_vec, [month_avg_vec[-1]]))
    avg_tm_stamp = np.concatenate(([first_tm_stamp], avg_tm_stamp, [np.max(month_tm_stamp_end)]))

    average_seasonality_series = np.interp(time_stamps, avg_tm_stamp, month_avg_vec)
    return average_seasonality_series

def tsEvaTransformSeriesToStationaryMultiplicativeSeasonality(timeStamps, series, timeWindow, *args, **kwargs):

    seasonalityTimeWindow = 2 * 30.4  # 2 months
    
    print('computing trend ...')
    class TrasfData:
        pass

    trasfData = TrasfData()

    statSeries, trendSeries, filledTimeStamps, filledSeries, nRunMn = tsEvaDetrendTimeSeries(timeStamps, series, timeWindow, *args, **kwargs)
    
    print('computing trend seasonality ...')
    trendSeasonality = tsEstimateAverageSeasonality(filledTimeStamps, statSeries)
    statSeries -= trendSeasonality
    
    print('computing slowly varying standard deviation ...')
    varianceSeries = tsEvaNanRunningVariance(statSeries, nRunMn)
    # further smoothing
    varianceSeries = tsEvaNanRunningMean(varianceSeries, int(np.ceil(nRunMn/2)))
    
    seasonalVarNRun = int(round(nRunMn / timeWindow * seasonalityTimeWindow))
    
    print('computing standard deviation seasonality ...')
    seasonalVarSeries = tsEvaNanRunningVariance(statSeries, seasonalVarNRun)
    seasonalStdDevSeries = np.sqrt(seasonalVarSeries / varianceSeries)
    seasonalStdDevSeries = tsEstimateAverageSeasonality(filledTimeStamps, seasonalStdDevSeries)
    
    stdDevSeriesNonSeasonal = np.sqrt(varianceSeries)
    
    # prevent division by zero
    stdDevSeriesNonSeasonal = np.where(stdDevSeriesNonSeasonal == 0, np.nan, stdDevSeriesNonSeasonal)
    seasonalStdDevSeries = np.where(seasonalStdDevSeries == 0, np.nan, seasonalStdDevSeries)
    
    statSeries /= (stdDevSeriesNonSeasonal * seasonalStdDevSeries)
    
    # Placeholder for running statistics
    # Assuming tsEvaNanRunningStatistics returns three moments
    # For simplicity, we can set them as NaNs or implement accordingly
    _, _, statSer3Mom, statSer4Mom = tsEvaNanRunningStatistics(statSeries, nRunMn)
    statSer3Mom = tsEvaNanRunningMean(statSer3Mom, int(np.ceil(nRunMn)))
    statSer4Mom = tsEvaNanRunningMean(statSer4Mom, int(np.ceil(nRunMn)))
    
    N = nRunMn
    # Error on the trend
    trendNonSeasonalError = np.nanmean(stdDevSeriesNonSeasonal) / np.sqrt(N)
    
    # Calculation of stdDevNonSeasonalError
    S = 2
    avgStdDev = np.nanmean(stdDevSeriesNonSeasonal)
    stdDevNonSeasonalError = avgStdDev * ( (2 * S ** 2) / (N ** 3) ) ** (1/4.)
    
    Ntot = len(series)
    trendSeasonalError = stdDevNonSeasonalError * np.sqrt(12 / Ntot + 1 / N)
    # Prevent negative or zero in root
    denom = Ntot ** 2 * N
    stdDevSeasonalError = seasonalStdDevSeries * (288.0 / denom) ** (1/4.)
    
    trendError = np.sqrt(trendNonSeasonalError ** 2 + trendSeasonalError ** 2)
    stdDevError = np.sqrt(
        (stdDevSeriesNonSeasonal * stdDevSeasonalError) ** 2 +
        (seasonalStdDevSeries * stdDevNonSeasonalError) ** 2
    )
    
    # Prepare output dictionary
    trasfData.runningStatsMulteplicity = nRunMn
    trasfData.stationarySeries = statSeries
    trasfData.trendSeries = trendSeries + trendSeasonality
    trasfData.trendSeriesNonSeasonal = trendSeries
    trasfData.stdDevSeries = stdDevSeriesNonSeasonal * seasonalStdDevSeries
    trasfData.stdDevSeriesNonSeasonal = stdDevSeriesNonSeasonal
    trasfData.trendNonSeasonalError = trendNonSeasonalError
    trasfData.stdDevNonSeasonalError = stdDevNonSeasonalError
    trasfData.trendSeasonalError = trendSeasonalError
    trasfData.stdDevSeasonalError = stdDevSeasonalError
    trasfData.trendError = trendError
    trasfData.stdDevError = stdDevError
    trasfData.timeStamps = filledTimeStamps
    trasfData.nonStatSeries = filledSeries
    trasfData.statSer3Mom = statSer3Mom
    trasfData.statSer4Mom = statSer4Mom
    
    return trasfData

def tsEvaNonStationary(timeAndSeries, timeWindow,**kwargs):

    transfType=kwargs.get('transfType','trend')
    minPeakDistanceInDays=kwargs.get('minPeakDistanceInDays',-1)
    ciPercentile=kwargs.get('ciPercentile',np.nan)
    potEventsPerYear=kwargs.get('potEventsPerYear',5)
    evdType=kwargs.get('evdType',['GEV', 'GPD'])
    gevType=kwargs.get('gevType','GEV')  # can be 'GEV' or 'Gumbel'
    
    for key, value in kwargs.items():
        if (key=='transfType'): 
            transfType=value
        if (key=='minPeakDistanceInDays'): 
            minPeakDistanceInDays=value
        if (key=='evdType'): 
            evdType=value
        if (key=='gevType'): 
            gevType=value
        if (key=='potEventsPerYear'): 
            potEventsPerYear=value
        if (key=='ciPercentile'): 
            ciPercentile=value

    if not(
            (
                (transfType == "trend" or transfType == "seasonal")
                or transfType == "trendCIPercentile"
            )
            or transfType == "seasonalCIPercentile"
    ): print("nonStationaryEvaJRCApproach: transfType can be in (trend, seasonal, trendCIPercentile)")
    
    if (minPeakDistanceInDays==-1):
        print("label parameter " "minPeakDistanceInDays" " must be set")
        
    nonStationaryEvaParams = []
    stationaryTransformData = []
	
    timeStamps = timeAndSeries[:, 0]
    series = timeAndSeries[:, 1]
    
    if (transfType=="trend"):
        print("evaluating long term variations of extremes")
        trasfData = tsEvaTransformSeriesToStationaryTrendOnly(timeStamps, series, timeWindow)
        
        gevMaxima = "annual"
        potEventsPerYear = 5
        
    elif (transfType=="seasonal"):
        print("evaluating long term an seasonal variations of extremes")
        trasfData = tsEvaTransformSeriesToStationaryMultiplicativeSeasonality(timeStamps, series, timeWindow, **kwargs);
        gevMaxima = "monthly"
        potEventsPerYear = 12
        
    elif (transfType=="trendCIPercentile"):
        if (ciPercentile==np.nan):
            print("For trendCIPercentile transformation the label parameter ''cipercentile'' is mandatory:")
            print("evaluating long term variations of extremes using the th percentile")
            gevMaxima = "annual"
            potEventsPerYear = 5
    elif (transfType=="seasonalCIPercentile"): 
        if (ciPercentile==np.nan):
            print("For seasonalCIPercentile transformation the label parameter ''cipercentile'' is mandatory")
            print(f"evaluating long term variations of extremes using the {ciPercentile:3f} th percentile")
            gevMaxima = "monthly"
            potEventsPerYear = 12
    
        # Adjust potEventsPerYear if overridden
    #if 'potEventsPerYear' in kwargs:
    #    potEventsPerYear = kwargs['potEventsPerYear']
    if potEventsPerYear != -1:
        potEventsPerYear = kwargs.get('potEventsPerYear', 5)
    

    ms = np.column_stack((trasfData.timeStamps, trasfData.stationarySeries))
    dt = tsEvaGetTimeStep(trasfData.timeStamps)
    minPeakDistance = minPeakDistanceInDays / dt
    
    # Estimate non stationary EVA parameters
    print("Executing stationary EVA")
    pointData = tsEvaSampleData(ms, meanEventsPerYear=potEventsPerYear, **kwargs)
    alphaCi = 0.68
    _, eva, is_valid = tsEVstatistics(pointData, alphaci=alphaCi, gevMaxima=gevMaxima, gevType=gevType, evdType=evdType, tail=None)
    if not is_valid:
        return None, None, False

    #    eva[1]['thresholdError'] = pointData['POT']['thresholdError']
    
    eva[1]['thresholdError'] = pointData['POT']['thresholdError']
    
    # GEV processing
    if eva[0]['parameters'] is not None:
        epsilonGevX = eva[0]['parameters']['epsilon']
        errEpsilonX = epsilonGevX - eva[0]['paramCIs']['epsilonci'][0]
        muGevX = eva[0]['parameters']['mu']
        errMuGevX = muGevX - eva[0]['paramCIs']['muci'][0]
        sigmaGevX = eva[0]['parameters']['sigma']
        errSigmaGevX = sigmaGevX - eva[0]['paramCIs']['sigci'][0]
        
        print("Transforming to non-stationary EVA...")
        epsilonGevNs = epsilonGevX
        errEpsilonGevNs = errEpsilonX
        sigmaGevNs = trasfData.stdDevSeries * sigmaGevX
        errSigmaGevFit = trasfData.stdDevSeries * errSigmaGevX
        errSigmaGevTransf = sigmaGevX * trasfData.stdDevError
        errSigmaGevNs = np.sqrt(errSigmaGevTransf**2 + errSigmaGevFit**2)
        
        muGevNs = trasfData.stdDevSeries * muGevX + trasfData.trendSeries
        errMuGevFit = trasfData.stdDevSeries * errMuGevX
        errMuGevTransf = np.sqrt((muGevX * trasfData.stdDevError)**2 + trasfData.trendError**2)
        errMuGevNs = np.sqrt(errMuGevTransf**2 + errMuGevFit**2)
        
        gevParams = {
            'epsilon': epsilonGevNs,
            'sigma': sigmaGevNs,
            'mu': muGevNs,
            'timeDelta': 365.25 if gevMaxima == 'annual' else 365.25 / 12,
            'timeDeltaYears': 1 if gevMaxima == 'annual' else 1 / 12,
        }
        
        gevParamErr = {
            'epsilonErr': errEpsilonGevNs,
            'sigmaErrFit': errSigmaGevFit,
            'sigmaErrTransf': errSigmaGevTransf,
            'sigmaErr': errSigmaGevNs,
            'muErrFit': errMuGevFit,
            'muErrTransf': errMuGevTransf,
            'muErr': errMuGevNs
        }
        
        gevObj = {
            'method': eva[0]['method'],
            'parameters': gevParams,
            'paramErr': gevParamErr,
            'stationaryParams': eva[0],
            'objs': {'monthlyMaxIndexes': pointData.get('monthlyMaxIndexes', None)}
        }
    else:
        gevObj = {
            'method': eva[0]['method'],
            'parameters': None,
            'paramErr': None,
            'stationaryParams': None,
            'objs': {'monthlyMaxIndexes': None}
        }
        

    # GPD processing
    if eva[1]['parameters'] is not None:
        epsilonPotX = eva[1]['parameters']['shape']
        errEpsilonPotX = epsilonPotX - eva[1]['paramCIs'][0][2]
        sigmaPotX = eva[1]['parameters']['sigma']
        errSigmaPotX = sigmaPotX - eva[1]['paramCIs'][0][0]
        thresholdPotX = eva[1]['parameters']['threshold']
        errThresholdPotX = eva[1]['thresholdError']
        nPotPeaks = eva[1]['parameters']['peaks']
        percentilePotX = eva[1]['parameters']['percentile']
        
        dtPeaks = minPeakDistance / 2
        dtPotX = (timeStamps[-1] - timeStamps[0]) / len(series) * dtPeaks

        epsilonPotNs = epsilonPotX
        errEpsilonPotNs = errEpsilonPotX
        sigmaPotNs = sigmaPotX * trasfData.stdDevSeries
        errSigmaPotFit = trasfData.stdDevSeries * errSigmaPotX
        errSigmaPotTransf = sigmaPotX * trasfData.stdDevError
        errSigmaPotNs = np.sqrt(errSigmaPotFit**2 + errSigmaPotTransf**2)
        
        thresholdPotNs = thresholdPotX * trasfData.stdDevSeries + trasfData.trendSeries
        thresholdErrFit = 0
        thresholdErrTransf = np.sqrt((trasfData.stdDevSeries * errThresholdPotX)**2 +(thresholdPotX * trasfData.stdDevError)**2 +trasfData.trendError**2)
        thresholdErr = thresholdErrTransf
        
        potParams = {
            'epsilon': epsilonPotNs,
            'sigma': sigmaPotNs,
            'threshold': thresholdPotNs,
            'percentile': percentilePotX,
            'timeDelta': dtPotX,
            'timeDeltaYears': dtPotX / 365.2425,
            'timeHorizonStart': np.min(trasfData.timeStamps),
            'timeHorizonEnd': np.max(trasfData.timeStamps),
            'nPeaks': nPotPeaks
        }
        
        potParamErr = {
            'epsilonErr': errEpsilonPotNs,
            'sigmaErrFit': errSigmaPotFit,
            'sigmaErrTransf': errSigmaPotTransf,
            'sigmaErr': errSigmaPotNs,
            'thresholdErrFit': thresholdErrFit,
            'thresholdErrTransf': thresholdErrTransf,
            'thresholdErr': thresholdErr
        }
        
        potObj = {
            'method': eva[1]['method'],
            'parameters': potParams,
            'paramErr': potParamErr,
            'stationaryParams': eva[1],
            'objs': {}
        }
    else:
        potObj = {
            'method': eva[1]['method'],
            'parameters': None,
            'paramErr': None,
            'stationaryParams': None,
            'objs': {}
        }
        
    # Final output
    del nonStationaryEvaParams
    nonStationaryEvaParams = [gevObj, potObj]
    stationaryTransformData = trasfData
    is_valid = True
    
    return nonStationaryEvaParams, stationaryTransformData, is_valid
    
def tsEvaStationary(time_and_series, **kwargs):
    # Default arguments
    minPeakDistanceInDays=kwargs.get('minPeakDistanceInDays',-1)
    potEventsPerYear=kwargs.get('potEventsPerYear',5)
    gevMaxima=kwargs.get('gevMaxima','annual')
    gevType=kwargs.get('gevType','GEV')  # can be 'GEV' or 'Gumbel'
    doSampleData=kwargs.get('doSampleData',True)
    potThreshold=kwargs.get('potThreshold',np.nan)
    evdType=kwargs.get('evdType',['GEV', 'GPD'])
    
    # Parse the named arguments (you can replace with your argument parser)
    for key, value in kwargs.items():
        if (key=='minPeakDistanceInDays'): 
            minPeakDistanceInDays=value
        if (key=='potEventsPerYear'): 
            potEventsPerYear=value
        if (key=='gevMaxima'): 
            gevMaxima=value
        if (key=='gevType'): 
            gevType=value
        if (key=='doSampleData'): 
            doSampleData=value
        if (key=='potThreshold'): 
            potThreshold=value
        if (key=='evdType'): 
            evdType=value

    tail="high"
    #minEventsPerYear=1
    if minPeakDistanceInDays == -1:
        raise ValueError("label parameter 'minPeakDistanceInDays' must be set")
    
    stationaryEvaParams = []
    
    print("Executing stationary EVA...")
    
    # Data sampling
    if doSampleData:
        # Removing NaN values
        time_and_series = time_and_series[~np.isnan(time_and_series[:, 1])]
                
        # Replace tsEvaSampleData with a custom data sampling function
        pointData = tsEvaSampleData(time_and_series, **kwargs)
    else:
        if np.isnan(potThreshold):
            raise ValueError("If doSampleData==False, you need to provide a value for the potThreshold.")
        pointData = tsTimeSeriesToPointData(time_and_series, potThreshold, 0)
    
    # Call to tsEVstatistics for GEV fitting
    evaAlphaCI = 0.68  # Approximation of 68% confidence interval
    EVmeta, EVdata, isValid = tsEVstatistics(pointData, alphaCI=evaAlphaCI, gevMaxima=gevMaxima, gevType=gevType, evdType=evdType)
    if not isValid:
        return None, False
    
    # GEV Parameters and Errors
    if ('GEV' in evdType):
        epsilonGevX = EVdata[0]['parameters']['epsilon']
        errEpsilonX = epsilonGevX - EVdata[0]['paramCIs']['epsilonci'][0]
        sigmaGevX = EVdata[0]['parameters']['sigma']
        errSigmaGevX = sigmaGevX - EVdata[0]['paramCIs']['sigci'][0]
        muGevX = EVdata[0]['parameters']['mu']
        errMuGevX = muGevX - EVdata[0]['paramCIs']['muci'][0]
            
        gevParams = {
            'epsilon': epsilonGevX,
            'sigma': sigmaGevX,
            'mu': muGevX,
            'timeDelta': 365.25 if gevMaxima == 'annual' else 365.25 / 12,
            'timeDeltaYears': 1 if gevMaxima == 'annual' else 1 / 12
        }
        gevParamStdErr = {
            'epsilonErr': errEpsilonX,
            'sigmaErr': errSigmaGevX,
            'muErr': errMuGevX
        }
        
        gevObj = {
            'method': EVdata[0]['method'],
            'parameters': gevParams,
            'paramErr': gevParamStdErr,
            'objs': {'monthlyMaxIndexes': pointData['monthlyMaxIndx']}
        }
    if ('GPD' in evdType):
        # Estimating the non-stationary GPD parameters
        epsilonPotX = EVdata[1]['parameters']['shape']
        errEpsilonPotX = epsilonPotX - EVdata[1]['paramCIs'][0][2]
        
        sigmaPotX = EVdata[1]['parameters']['sigma']
        errSigmaPotX = sigmaPotX - EVdata[1]['paramCIs'][0][0]
        thresholdPotX = EVdata[1]['parameters']['threshold']
        errThresholdPotX = pointData['POT']['thresholdError']
        percentilePotX = EVdata[1]['parameters']['percentile']
        nPotPeaks = EVdata[1]['parameters']['peaks']

        
        timeStamps = time_and_series[:, 0]
        dt = tsEvaGetTimeStep(timeStamps)
        minPeakDistance = minPeakDistanceInDays / dt
        dtSample = (timeStamps[-1] - timeStamps[0]) / len(timeStamps)
        dtPotX = max(dtSample, dtSample * minPeakDistance)

        potParams = {
            'epsilon': epsilonPotX,
            'sigma': sigmaPotX,
            'threshold': thresholdPotX,
            'percentile': percentilePotX,
            'timeDelta': dtPotX,
            'timeDeltaYears': dtPotX / 365.2425,
            'timeHorizonStart': np.min(timeStamps),
            'timeHorizonEnd': np.max(timeStamps),
            'nPeaks': nPotPeaks
        }

        potParamStdErr = {
            'epsilonErr': errEpsilonPotX,
            'sigmaErr': errSigmaPotX,
            'thresholdErr': errThresholdPotX
        }
        
        potObj = {
            'method': EVdata[1]['method'],
            'parameters': potParams,
            'paramErr': potParamStdErr,
            'objs': []
        }
        
    stationaryEvaParams={0:gevObj,1:potObj}
        
    return stationaryEvaParams

def log_likelihood(params, data):
    shape, loc, scale = params
    return np.sum(genpareto.logpdf(data, shape, loc, scale))

#def tsEVstatistics(pointData, alphaCI=0.95, gevMaxima='annual', gevType='GEV', evdType=['GEV', 'GPD'], shape_bnd=[-0.5, 1]):
#def tsEVstatistics(pointData, alphaCI=0.95, gevMaxima='annual', gevType='GEV', evdType=['GEV', 'GPD'], shape_bnd=[-0.5, 1]):
def tsEVstatistics(pointData, **kwargs):
    
    # Create empty data structures
    EVmeta = {}
    EVdata = {}
    isValid = True

    minvars=1
    minGEVSample = 10

    alphaCI=kwargs.get('alphaCI',0.95)
    gevMaxima=kwargs.get('gevMaxima','annual')
    gevType=kwargs.get('gevType','GEV')
    evdType=kwargs.get('evdType',['GEV', 'GPD'])

    for key, value in kwargs.items():
        if (key=='alphaCI'): 
            alphaCI=value
        if (key=='gevMaxima'): 
            gevMaxima=value
        if (key=='gevType'): 
            gevType=value
        if (key=='evdType'): 
            evdType=value
        

    # Define Tr vector
    Tr = [5, 10, 20, 50, 100, 200, 500, 1000]
    EVmeta['Tr'] = Tr
    nyears = len(pointData['annualMax'])

    imethod = 1
    methodname = 'GEVstat'
    paramEsts = np.zeros(3)
    paramCIs = np.full((2, 3), np.nan)
    rlvls = np.zeros(len(Tr))

    # GEV statistics
    if (('GEV' in evdType) and pointData['annualMax'] is not None):
        if gevMaxima == 'annual':
            tmpmat = np.array(pointData['annualMax'])
        elif gevMaxima == 'monthly':
            tmpmat = np.array(pointData['monthlyMax'])
        else:
            raise ValueError(f'Invalid gevMaxima type: {gevMaxima}')

        iIN = ~np.isnan(tmpmat)
        
        if sum(iIN) >= minGEVSample:
            tmp = tmpmat[iIN]
            # Perform GEV/Gumbel fitting and computation of return levels
            if gevType == "GEV":
                stdfit = True
                # Try to fit GEV with bounded shape parameters and stderr, reduces the constraints if no fit.
                try:
#                    fit  = gev.fit(tmp, method="MLE",loc=np.mean(tmp),scale=np.std(tmp))
                    gev_instance = Bootstrap_fit(tmp)
#                    params,gev_confidence_interval = gev_instance.fit_genextreme()
                    params,paramCL = gev_instance.fit_genextreme()
                    paramEsts = {'epsilon': params[0], 'mu': params[1], 'sigma': params[2]}
                    paramCIs = {'epsilonci': paramCL[0],'muci': paramCL[1], 'sigci': paramCL[2]}

                    alphaCIx = 1 - alphaCI
                    
                except Exception as e:
                    stdfit = False
                    print("Not able to fit GEV stderr")
                        
            elif gevType == "Gumbel":
                gumbel_instance = Bootstrap_fit(tmp)
                params,paramCL = gumbel_instance.fit_gumbel()

                paramEsts = {'epsilon': 0, 'mu': params[0], 'sigma': params[1]}
                paramCIs = {'epsilonci': [0,0],'muci': paramCL[0], 'sigci': paramCL[1]}
                alphaCIx = 1 - alphaCI
                
            else:    
                print("tsEVstatistics: invalid gevType ",gevType," Can be only GEV or Gumbel")
                methodname = 'No fit'
                rlvls = None
                paramEsts = None
                paramCIs = None
                
    Tr_inv=  [1 - 1 / x for x in Tr]  
    if gevType == "GEV":
        rlvls = gev.ppf(Tr_inv, c=-paramEsts['epsilon'], loc=paramEsts['mu'], scale=paramEsts['sigma'])
    if gevType == "Gumbel":
        rlvls = gumbel_r.ppf(Tr_inv, loc=paramEsts['mu'], scale=paramEsts['sigma'])

    EVdata[0] = {
        'method': methodname,
        'values': rlvls,
        'parameters': paramEsts,
        'paramCIs': paramCIs
    }
    
    # Create output structures for GEV statistics
    # GPD statistics
    imethod = 2
    methodname = 'GPDstat'
    paramEsts = np.zeros(6)
    paramCIs = np.full((2, 3), np.nan)
    rlvls = np.zeros(len(Tr))
    if ('GPD' in evdType) and (pointData['annualMax'] is not None):
        # Perform GPD fitting and computation of return levels
        ik = 1
        d1 = pointData['POT']['peaks']-pointData['POT']['threshold']
        gpd_instance = Bootstrap_fit(d1)
        paramEsts,paramCIs,se = gpd_instance.fit_genpareto()
        alphaCIx = 1 - alphaCI

        ksi = paramEsts[0] # shape
        sgm = paramEsts[2] # scale
        
        if (ksi < -.5):
            probs = [alphaCIx / 2, 1 - alphaCIx / 2]
            
            kci = norm.ppf(probs, ksi, se[2])
            kci[kci < -1] = -1
        
            # Compute the CI for sigma using a normal approximation for log(sigmahat)
            # and transform back to the original scale.
            lnsigci = norm.ppf(probs, np.log(sgm), se[2] / sgm)
            paramCIs = [kci, np.exp(lnsigci)]
        
        # Create output structures for GPD statistics
        # Assign values to paramEstsall
        paramEstsall = {'sigma':sgm, 'shape': ksi, 'threshold': pointData['POT']['threshold'], 'length':len(d1), 'peaks':len(pointData['POT']['peaks']), 'percentile':pointData['POT']['percentile']}
        
        # Assign values to rlvls
        for i in range(len(Tr)):
            rlvls[i] = pointData['POT']['threshold'] + (sgm/ksi) * ((((len(d1)/len(pointData['POT']['peaks']))*(1/Tr[i]))**(-ksi))-1)

        EVdata[1] = {
            'method': methodname,
            'values': rlvls,
            'parameters': paramEstsall,
            'paramCIs': np.flip(paramCIs,1)
            }
    else:
        methodname = 'No fit'
        ik = 1
        th = pointData['POT']['threshold']
        d1 = pointData['POT']['peaks']
        paramEstsall = [pointData['POT']['pars'][0], pointData['POT']['pars'][1],
                        pointData['POT']['threshold'], len(d1),
                        len(pointData['POT']['peaks']), pointData['POT']['percentile']]
        EVdata[1] = {
            'method': methodname,
            'values': None,
            'parameters': paramEstsall,
            'paramCIs': None
        }
        print("could not estimate GPD: bounded distribution")
        
        # Return outputs
    return EVmeta, EVdata, isValid

def tsGetNumberPerYear(ms, locs):
    
    # Get all years
    sdfull = np.arange(np.nanmin(ms[:, 0]), np.nanmax(ms[:, 0]))
 
    # Make full time vector
    sdfull2 = sdfull - min(sdfull)

    # Round to years
    sdfull2_dt = pd.to_datetime(sdfull2, utc=True, unit='D', origin='unix')
    yearss = np.unique(sdfull2_dt.year-1970)
    
    sdser=ms[:,0] - min(ms[:,0])
    sdser_dt = pd.to_datetime(sdser, utc=True, unit='D', origin='unix')
    yearsser, icmser = np.unique(sdser_dt.year-1970, return_inverse=True)

    sdp = ms[locs, 0]

    sdp_dt = pd.to_datetime(sdp-min(sdfull), utc=True, unit='D', origin='unix')
    yearssp, icm = np.unique(sdp_dt.year-1970, return_inverse=True)

    # Prepare the series time vector
    df1 = pd.DataFrame({'icm': icm, 'sdp': sdp})
    
    ecoount = df1.groupby('icm').size().values
    
    df2 = pd.DataFrame({'icmser': icmser, 'sdser': sdser})
    ecoountSeries = df2.groupby('icmser').size().values
    
    iexcl=ecoountSeries/max(ecoountSeries)<0.5;
    
    nperYear = np.full(len(yearss), np.nan)
    nperYear[yearsser]=0;
    nperYear[yearssp]=ecoount;
    for i in range(len(iexcl)):
        if iexcl[i]==True:
            nperYear[yearsser[i]]=np.nan;
    
    return nperYear

def tsEvaComputeMonthlyMaxima(time_and_series):
    time_stamps = time_and_series[:, 0]
    tmvec = pd.to_datetime(np.array(time_stamps) - 719529, unit='D', origin='unix')
    srs = time_and_series[:, 1]

    yrs = tmvec.year
    mnts = tmvec.month
    mnttmvec = pd.DataFrame({'yrs': yrs, 'mnts': mnts})
    vals_indxs = np.arange(0, len(srs))
    
    monthly_max_indx = mnttmvec.groupby(['yrs', 'mnts']).apply(lambda x: find_max(x.index, srs), include_groups=False).reset_index(name='valsIndxs')
    monthly_max_indx['valsIndxs'] = monthly_max_indx['valsIndxs'].astype(int)
    monthly_max_indx = monthly_max_indx.sort_values(by=['yrs', 'mnts'])['valsIndxs']
    monthly_max = srs[monthly_max_indx]
    monthly_max_date = time_stamps[monthly_max_indx]

    ret=pd.DataFrame({'monthlyMax': monthly_max, 'monthlyMaxDate': monthly_max_date, 'monthlyMaxIndx': monthly_max_indx})
    return ret

def tsEvaComputeAnnualMaxima(time_and_series):
    time_stamps = time_and_series[:, 0]
    tmvec = pd.to_datetime(np.array(time_stamps) - 719529, unit='D', origin='unix')
    srs = time_and_series[:, 1]
    years = tmvec.year
    srs_indices = range(0, len(srs))
    unique_years = np.unique(years)
    df = pd.DataFrame({'years': years, 'srs_indices': srs_indices, 'srs': srs})
    annual_max_indx = df.groupby('years').apply(lambda group: find_max(group['srs_indices'].values, df['srs'].values), include_groups=False)
    annual_max = [srs[i] for i in annual_max_indx]
    annual_max_date = time_stamps[annual_max_indx]

    ret=pd.DataFrame({'annualMax': annual_max, 'annualMaxDate': annual_max_date, 'annualMaxIndx': annual_max_indx})
    return ret

def find_max(indices, srs):
    if len(indices) == 0:
        return None
    return indices[np.argmax(srs[indices])] 



def declust(data, threshold, minrundistance):
    # Identify the peaks that exceed the threshold
    data=np.array(data)
    peaks = data > threshold
    n = len(data)
    cnr=0
    en=0
    ien=0
    thExceedances = []
    clusters = []
    sizes = []
    cluster_peaks = []
    clusterMaxima = []
    isClusterMax = []
    exceedanceTimes = []
    InterExceedTimes = []
    InterCluster = []
    intcl = []
    
    last_peak = 0  # Initialize to ensure the first peak is included
    for i in range(n):
        if (i >= minrundistance-1):
            ien = ien + 1
        if peaks[i]:
            if (en > 0):
                InterExceedTimes.append(ien)
                ien=0
            thExceedances.append(data[i])
            exceedanceTimes.append(i+1)
            if (i - last_peak > minrundistance):
                sizes.append(en)
                cnr = cnr + 1
                en=0
            last_peak = i
            en = en + 1
            clusters.append(cnr+1);
    cnr = cnr+1

    n=len(clusters)-1
    for i in range(n):
        cluster_peaks.append(thExceedances[i])
        if clusters[i+1] > clusters[i]:
            clusterMaxima.append(max(cluster_peaks))
            cluster_peaks=[]
            i=i-1
        elif i==n-1:
            cluster_peaks.append(thExceedances[i+1])
            clusterMaxima.append(max(cluster_peaks))
            
    j=0
    for i in range(len(thExceedances)-1):
        if (InterExceedTimes[i]>minrundistance):
            InterCluster.append(True)
        else:
            InterCluster.append(False)
        if (j<len(clusterMaxima)):
            if (thExceedances[i]==clusterMaxima[j]):
                isClusterMax.append(True)
                j = j+1
            else:
                isClusterMax.append(False)
        else:
            isClusterMax.append(False)
    isClusterMax.append(False)

    peakev=pd.DataFrame({
        'thExceedances': pd.Series(thExceedances),
        'clusters': pd.Series(clusters),
        'sizes': pd.Series(sizes),
        'clusterMaxima': pd.Series(clusterMaxima),
        'isClusterMax': pd.Series(isClusterMax),
        'exceedanceTimes': pd.Series(exceedanceTimes),
        'InterExceedTimes': pd.Series(InterExceedTimes),
        'InterCluster': pd.Series(InterCluster)
    })
    
    return peakev

def declustpeaks(data, minpeakdistance=10, minrundistance=7, qt=None):
     pks, _ = find_peaks(data, distance=minpeakdistance, height=qt)
     peakev = declust(data, threshold=qt, minrundistance=minrundistance)
     
     intcl = [True] + peakev['InterCluster'].to_list()
     intcl.pop()
     
    
     peakex = pd.DataFrame({
          'Qs': peakev['thExceedances'],
          'Istart': intcl,
          'clusters': peakev['clusters'],
          'IsClustermax': peakev['isClusterMax'],
          'exceedances': peakev['exceedanceTimes']
     })
     
     evmax = peakex[peakex['IsClustermax']==True]
     ziz = peakex.groupby('clusters')['exceedances'].apply(lambda x: x.max() - x.min() + 1).reset_index()
     st = peakex.groupby('clusters')['exceedances'].apply(lambda x: x.min()).reset_index()
     end = peakex.groupby('clusters')['exceedances'].apply(lambda x: x.max()).reset_index()    
     evmax['dur']= pd.Series(ziz['exceedances'].values,index=evmax.index)
     evmax['durx']= pd.Series(peakev['sizes'].dropna().values,index=evmax.index)
     
     peakdt = pd.DataFrame({
          'Q': evmax['Qs'],
          'max': evmax['exceedances'],
          'cluster': evmax['clusters']
     })
     peakdt['start']=pd.Series(st['exceedances'].values,index=peakdt.index)
     peakdt['end']=pd.Series(end['exceedances'].values,index=peakdt.index)
     peakdt['dur']=pd.Series(ziz['exceedances'].values,index=peakdt.index)
     peakdt = peakdt.sort_values(by=['Q'], ascending=False)
     return peakdt


def empdis(x, nyr):
    ts = list(range(1, len(x) + 1))
    ts = pd.DataFrame(ts, columns=['ts']).sort_values(by=0,axis=1).reset_index(drop=True)
    x = pd.DataFrame(x, columns=['x']).sort_values(by=0,axis=1).reset_index(drop=True)
    
    epyp = len(x) / nyr
    rank = x.rank(axis=0, method='min', na_option='keep')
    hazen = (rank - 0.5) / len(x)
    cunnane = (rank - 0.4) / (len(x) + 0.2)
    gumbel = -np.log(-np.log(hazen))
    n = len(x)
    rpc = (1 / (1 - np.arange(1, n + 1) / (n + 1))) / 12
    nasmp = x.apply(lambda col: col.count(), axis=0)
    epdp = rank / np.repeat(nasmp+1,rank.shape[1])
    empip = pd.DataFrame({
        'emp.RP': (1 / (epyp * (1 - epdp))).to_numpy().flatten(),
        'haz.RP': (1 / (epyp * (1 - hazen))).to_numpy().flatten(),
        'cun.RP': (1 / (epyp * (1 - cunnane))).to_numpy().flatten(),
        'gumbel': (gumbel).to_numpy().flatten(),
        'emp.f': (epdp).to_numpy().flatten(),
        'emp.hazen': (hazen).to_numpy().flatten(),
        'emp.cunnane': (cunnane).to_numpy().flatten(),
        'Q': (x['x']).to_numpy().flatten(),
        'timestamp': (ts['ts']).to_numpy().flatten()
    })
    return empip

def empdisl(x, nyr):
    # Convert the input to a pandas DataFrame and sort it in decreasing order
    x = pd.DataFrame(np.sort(x, axis=0)[::-1])
    
    # Calculate the event per year (epyp)
    epyp = len(x) / nyr
    
    # Calculate rank
    rank = x.rank(axis=0, method='min', na_option='keep', ascending=False)
    
    # Calculate Hazen
    hazen = (rank - 0.5) / len(x)
    
    # Calculate Gumbel
    gumbel = -np.log(-np.log(hazen))
    
    # Calculate number of non-missing values (nasmp)
    nasmp = np.sum(~np.isnan(x.values), axis=0)
    
    # Calculate Empirical Distribution Probability (epdp)
    epdp = rank / (nasmp + 1)
    
    # Create the empip DataFrame with calculated columns
    empip = pd.DataFrame({
        "emp.RP": 1 / (epyp * (1 - epdp)).to_numpy().flatten(),
        "haz.RP": 1 / (epyp * (1 - hazen)).to_numpy().flatten(),
        "gumbel": gumbel.to_numpy().flatten(),
        "emp.f": epdp.to_numpy().flatten(),
        "emp.hazen": hazen.to_numpy().flatten(),
        "Q": x.to_numpy().flatten() 
    })
    
    return empip
