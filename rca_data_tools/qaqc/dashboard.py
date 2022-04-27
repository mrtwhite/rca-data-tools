# -*- coding: utf-8 -*-
#import matplotlib
#matplotlib.rcParams['backend'] = 'TkAgg'
#matplotlib.use("TkAgg")

import ast
from datetime import datetime, timedelta
from dateutil import parser
import gc
import io
import json
import numpy as np
import pandas as pd
import requests
import s3fs
import statistics as st
import xarray as xr


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean # noqa
from scipy.interpolate import griddata

def pressureBracket(pressure, clim_dict):
    bracketList = []
    pressBracket = 'notFound'

    for bracket in clim_dict['1'].keys():
        bracketList.append(ast.literal_eval(bracket))
    if pressure < bracketList[0][0]:
        pressBracket = bracketList[0]
    elif pressure > bracketList[-1][1] - 1:
        pressBracket = bracketList[-1]
    else:
        for bracket in bracketList:
            if (pressure >= bracket[0]) & (pressure < bracket[1]):
                pressBracket = bracket
                break

    return pressBracket


def extractClim(timeRef, profileDepth, overlayData_clim):

    depth = float(profileDepth)
    climBracket = pressureBracket(depth, overlayData_clim)
    climTime = []
    climMinus3std = []
    climPlus3std = []
    climData = []

    if 'notFound' in climBracket:
        climInterpolated_hour = pd.DataFrame()
    else:
        for i in range(1, 13):
            climMonth = i
            climatology = ast.literal_eval(
                overlayData_clim[str(climMonth)][str(climBracket)]
            )
            # current year
            climTime.append(datetime(timeRef.year, i, 15))
            climMinus3std.append(climatology[0])
            climPlus3std.append(climatology[1])
            climData.append(st.mean([climatology[0], climatology[1]]))
            # extend climatology to previous year
            climTime.append(datetime(timeRef.year - 1, i, 15))
            climMinus3std.append(climatology[0])
            climPlus3std.append(climatology[1])
            climData.append(st.mean([climatology[0], climatology[1]]))
            # extend climatology to next year
            climTime.append(datetime(timeRef.year + 1, i, 15))
            climMinus3std.append(climatology[0])
            climPlus3std.append(climatology[1])
            climData.append(st.mean([climatology[0], climatology[1]]))

        zipped = zip(climTime, climMinus3std, climPlus3std, climData)
        zipped = list(zipped)
        sortClim = sorted(zipped, key=lambda x: x[0])

        climSeries = pd.DataFrame(
            sortClim,
            columns=['climTime', 'climMinus3std', 'climPlus3std', 'climData'],
        )
        climSeries.set_index(['climTime'], inplace=True)

        upsampled_hour = climSeries.resample('H')
        climInterpolated_hour = upsampled_hour.interpolate(method='linear')

    return climInterpolated_hour



def gridProfiles(ds,pressureName,variableName,profileIndices):

    mask = (profileIndices['start'] > ds.time[0].values) & (profileIndices['end'] <= ds.time[-1].values)
    profileIndices = profileIndices.loc[mask]

    if profileIndices.empty:
        gridX = np.zeros(1)
        gridY = np.zeros(1)
        gridZ = np.zeros(1)

    else:
        profileIndices = profileIndices.reset_index()

        descentSamples = ['pco2_seawater','ph_seawater']

        if variableName in descentSamples:
            start = 'peak'
            end = 'end'
            invert = False
        else:
            start = 'start'
            end = 'peak'
            invert = True

        gridX = np.zeros(len(profileIndices))
        gridY = np.arange(0, 190, 0.5) ### grid points every 0.5 meters
        gridZ = np.zeros((len(gridY),len(gridX)))
        for index, row in profileIndices.iterrows():
            startTime = row[start]
            endTime = row[end]
            #gridX[index] = row['peak'].timestamp()
            ds_sub = ds.sel(time=slice(startTime,endTime))
            if len(ds_sub['time']) > 0: 
                gridX[index] = row['peak'].timestamp()
                #gridX[index] = row['peak'].timestamp()
                if invert:
                    variable = np.flip(ds_sub[variableName].values)
                    pressure = np.flip(ds_sub[pressureName].values)
                else:
                    variable = ds_sub[variableName].values
                    pressure = ds_sub[pressureName].values
                try:
                    profile = np.interp(gridY,pressure,variable)
                    gridZ[:,index] = profile
                except:
                    gridZ[:,index] = np.nan
                    #print('setting gridZ to nan because interp failed...')
                ### TODO: fill grid with nans outside of pressure values in variable
            else:
                gridZ[:,index] = np.nan
                #print('setting gridZ to nan because timeslice was empty')
    #print(gridZ)
    #print(gridX)
    return(gridX,gridY,gridZ)

    
def loadDeploymentHistory(refDes):
    deployHistory = {}
    (site, node, sensor1, sensor2) = refDes.split('-')
    dateColumns = ['startDateTime','stopDateTime']
    gh_baseURL = 'https://raw.githubusercontent.com/oceanobservatories/asset-management/master/deployment/'
    deployURL = gh_baseURL + site + '_Deploy.csv'
    
    download = requests.get(deployURL)
    if download.status_code == 200:
        df = pd.read_csv(io.StringIO(download.content.decode('utf-8')),parse_dates=dateColumns)
        df_sort = df.sort_values(by=["Reference Designator","startDateTime"],ascending=False)
        for i in df_sort['Reference Designator'].unique():
            deployHistory[i] = [{'deployDate':df_sort['startDateTime'][j],'deployEnd':df_sort['stopDateTime'][j],
                                'deployNum':df_sort['deploymentNumber'][j]} 
                                for j in df_sort[df_sort['Reference Designator']==i].index]

        
    else:
        logger.warning(f"error retrieving deployment history for {site}")

    return deployHistory



def loadProfiles(refDes):

    profileList = []
    dateColumns = ['start','peak','end']
    (site, node, sensor1, sensor2) = refDes.split('-')
    gh_baseURL = 'https://raw.githubusercontent.com/OOI-CabledArray/profileIndices/main/'
    profiles_URL = gh_baseURL + site + '_profiles.csv'
    download = requests.get(profiles_URL)
    if download.status_code == 200:
        profileList = pd.read_csv(io.StringIO(download.content.decode('utf-8')),parse_dates=dateColumns)
    else:
        logger.warning(
            f"error retrieving profileIndices for {site}"
        )
    
    return profileList

def loadQARTOD(refDes, param, sensorType, logger=None):
    if logger is None:
        from loguru import logger

    (site, node, sensor1, sensor2) = refDes.split('-')
    sensor = sensor1 + '-' + sensor2

    # Load climatology and gross range values

    githubBaseURL = 'https://raw.githubusercontent.com/oceanobservatories/qc-lookup/master/qartod/'
    clim_URL = (
        githubBaseURL
        + sensorType
        + '/climatology_tables/'
        + refDes
        + '-'
        + param
        + '.csv'
    )
    grossRange_URL = (
        githubBaseURL
        + sensorType
        + '/'
        + sensorType
        + '_qartod_gross_range_test_values.csv'
    )

    download = requests.get(grossRange_URL)
    if download.status_code == 200:
        df_grossRange = pd.read_csv(
            io.StringIO(download.content.decode('utf-8'))
        )
        qcConfig = df_grossRange.qcConfig[
            (df_grossRange.subsite == site)
            & (df_grossRange.node == node)
            & (df_grossRange.sensor == sensor)
        ]
        qcConfig_json = qcConfig.values[0].replace("'", "\"")
        grossRange_dict = json.loads(qcConfig_json)
    else:
        logger.warning(
            f"error retrieving gross range data for {refDes} {param} {sensorType}"
        )
        grossRange_dict = {}

    download = requests.get(clim_URL)
    if download.status_code == 200:
        df_clim = pd.read_csv(io.StringIO(download.content.decode('utf-8')))
        climRename = {
            'Unnamed: 0': 'depth',
            '[1, 1]': '1',
            '[2, 2]': '2',
            '[3, 3]': '3',
            '[4, 4]': '4',
            '[5, 5]': '5',
            '[6, 6]': '6',
            '[7, 7]': '7',
            '[8, 8]': '8',
            '[9, 9]': '9',
            '[10, 10]': '10',
            '[11, 11]': '11',
            '[12, 12]': '12',
        }

        df_clim.rename(columns=climRename, inplace=True)
        clim_dict = df_clim.set_index('depth').to_dict()
    else:
        logger.warning(
            f"error retrieving climatology data for {refDes} {param} {sensorType}"
        )
        clim_dict = {}

    return (grossRange_dict, clim_dict)

def loadStatus():
    statusResponse = requests.get("https://nereus.ooirsn.uw.edu/api/public/v1/instruments/operational-status").text
    status = json.loads(statusResponse)

    return status


def loadData(site, sites_dict):
    fs = s3fs.S3FileSystem(anon=True)
    zarrDir = 'ooi-data/' + sites_dict[site]['zarrFile']
    zarr_store = fs.get_mapper(zarrDir)
    # TODO: only request parameters listed in sites_dict[site][dataParameters]?
    # requestParams = sites_dict[site]['dataParameters'].strip('"').split(',')
    ds = xr.open_zarr(zarr_store, consolidated=True)

    return ds

def listDeployTimes(deployDict):
    deployTimes = []
    for deploy in deployDict:
        deployTimes.append(deploy['deployDate'])
        
    return deployTimes

def plotProfilesGrid(
    Yparam,
    pressParam,
    paramData,
    plotTitle,
    zLabel,
    timeRef,
    yMin,
    yMax,
    zMin,
    zMax,
    colorMap,
    fileName_base,
    overlayData_clim,
    overlayData_near,
    span,
    spanString,
    profileList,
    statusDict,
    site,
    ):
    ### QC check for grid...this will be replaced with a new range for "gross range"
    if 'pco2' in Yparam:
        paramData = paramData.where((paramData[Yparam] < 2000), drop=True)
    #if 'par' in Yparam:
    #    paramData = paramData.where((paramData[Yparam] > 0) & (paramData[Yparam] < 2000), drop=True)

    # Initiate fileName list
    fileNameList = []

    # Plot Overlays
    overlays = ['clim', 'near', 'time', 'none']

    # Data Ranges
    ranges = ['full', 'local']

    lineColors = [
        '#1f78b4',
        '#a6cee3',
        '#b2df8a',
        '#33a02c',
        '#fb9a99',
        '#e31a1c',
        '#fdbf6f',
        '#ff7f00',
    ]
    balanceBig = plt.get_cmap('cmo.balance', 512)
    balanceBlue = ListedColormap(balanceBig(np.linspace(0, 0.5, 256)))

    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')

    statusString = statusDict[site]
    statusColors = {'OPERATIONAL': 'green',
                'FAILED': 'red',
                'TROUBLESHOOTING': 'red',
                'OFFLINE': 'blue',
                'UNCABLED': 'blue',
                'DATA_QUALITY': 'red',
                'NOT_DEPLOYED': 'blue'
                }



    def plotter(Xx,Yy,Zz,plotType,colorBar,annotation,params):

        plt.close('all')
        plt.rcParams["font.family"] = "serif"

        fig, ax = plt.subplots()
        fig.set_size_inches(5, 1.75)
        fig.patch.set_facecolor('white')
        plt.title(plotTitle, fontsize=4, loc='left')
        plt.title(statusString, fontsize=4, fontweight=0, color=statusColors[statusString], loc='right', style='italic' )
        plt.ylabel('Pressure (dbar)', fontsize=4)
        ax.tick_params(direction='out', length=2, width=0.5, labelsize=4)
        ax.ticklabel_format(useOffset=False)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats = [
            '%y',  # ticks are mostly years
            '%b',  # ticks are mostly months
            '%m/%d',  # ticks are mostly days
            '%H h',  # hrs
            '%H:%M',  # min
            '%S.%f',
        ]  # secs
        formatter.zero_formats = [
            '',  # ticks are mostly years, no need for zero_format
            '%b-%Y',  # ticks are mostly months, mark month/year
            '%m/%d',  # ticks are mostly days, mark month/year
            '%m/%d',  # ticks are mostly hours, mark month and day
            '%H',  # ticks are montly mins, mark hour
            '%M',
        ]  # ticks are mostly seconds, mark minute

        formatter.offset_formats = [
            '',
            '',
            '',
            '',
            '',
            '',
        ]

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(False)
        ax.invert_yaxis()
        plt.xlim(xMin, xMax)
    
        if 'contour' in plotType:
            if 'local' in params['range']:
                colorRange = params['vmax'] - params['vmin']
                cbarticks = np.arange(params['vmin'],params['vmax'],colorRange/50)
                graph = ax.contourf(Xx, Yy, Zz, cbarticks, cmap=colorBar)
            else:
                graph = ax.contourf(Xx, Yy, Zz, 50, cmap=colorBar)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            cbar = plt.colorbar(graph, cax=cax)
            if 'local' in params['range']:
                graph.set_clim(params['vmin'], params['vmax'])
            cbar.update_ticks()
            cbar.formatter.set_useOffset(False)
            cbar.ax.set_ylabel(zLabel, fontsize=4)
            cbar.ax.tick_params(length=2, width=0.5, labelsize=4)
            
        
        if 'empty' in plotType:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            for axis in ['top','bottom','left','right']:
                cax.spines[axis].set_linewidth(0)
            cax.set_xticks([])
            cax.set_yticks([])
            plt.annotate(annotation, xy=(0.3, 0.5), xycoords='figure fraction')

        if 'clim' in plotType:
            colorRange = params['vmax'] - params['vmin']
            cbarticks = np.arange(params['vmin'],params['vmax'],colorRange/50)
            if 'yes' in params['norm']:
                divnorm = colors.TwoSlopeNorm(
                    vmin=params['vmin'], vcenter=0,vmax=params['vmax']
                    )
                graph = ax.contourf(Xx, Yy, Zz, cbarticks, cmap=colorBar,vmin=params['vmin'],
                                vmax=params['vmax'],norm=divnorm)
            else:
                graph = ax.contourf(Xx, Yy, Zz, cbarticks, cmap=colorBar,vmin=params['vmin'],vmax=params['vmax'])
            m = ScalarMappable(cmap=graph.get_cmap())
            m.set_array(graph.get_array())
            m.set_clim(graph.get_clim())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            cbar = plt.colorbar(graph, cax=cax)
            cbar.update_ticks()
            cbar.formatter.set_useOffset(False)
            cbar.ax.set_ylabel(zLabel, fontsize=4)
            cbar.ax.tick_params(length=2, width=0.5, labelsize=4)
        
        return fig

    print('plotting grid for timeSpan: ', span)

    if 'deploy' in spanString:
        deployHistory = loadDeploymentHistory(site)
        deployTimes = listDeployTimes(deployHistory[site])

        timeRef_deploy = deployTimes[0]
        startDate = timeRef_deploy - timedelta(days=15)
        endDate = timeRef_deploy + timedelta(days=15)
        xMin = startDate - timedelta(days=15 * 0.002)
        xMax = endDate + timedelta(days=15 * 0.002)
    else:
        endDate = timeRef
        startDate = timeRef - timedelta(days=int(span))
        xMin = startDate - timedelta(days=int(span) * 0.002)
        xMax = endDate + timedelta(days=int(span) * 0.002)

    baseDS = paramData.sel(time=slice(startDate, endDate))
    ### drop nans from dataset
    baseDS = baseDS.where( (baseDS[Yparam].notnull()) & (baseDS[pressParam].notnull()), drop=True)
    scatterX = baseDS.time.values
    scatterY = np.array([])
    scatterZ = np.array([])
    if len(scatterX) > 5:
        scatterY = baseDS[pressParam].values
        scatterZ = baseDS[Yparam].values
        # create interpolation grid
        xMinTimestamp = xMin.timestamp()
        xMaxTimestamp = xMax.timestamp()
        if profileList.empty:
            print('profileList empty...interpolating with old method...')
            # x grid in seconds, with points every 1 hour (3600 seconds)
            xi_arr = np.arange(xMinTimestamp, xMaxTimestamp, 3600)
            yi_arr = np.arange(yMin, yMax, 0.5)
            xi, yi = np.meshgrid(xi_arr, yi_arr)

            scatterX_TS = [((dt64 - unix_epoch) / one_second) for dt64 in scatterX]

            # interpolate data to grid
            zi = griddata(
                (scatterX_TS, scatterY), scatterZ, (xi, yi), method='linear'
            )

            xiDT = xi.astype('datetime64[s]')
            # mask out any time gaps greater than 1 day
            timeGaps = np.where(np.diff(scatterX_TS) > 86400)
            if len(timeGaps[0]) > 1:
                gaps = timeGaps[0]
                for gap in gaps:
                    gapMask = (xi > scatterX_TS[gap]) & (xi < scatterX_TS[gap + 1])
                    zi[gapMask] = np.nan
        else:
            xi_arr, yi_arr, zi = gridProfiles(baseDS,pressParam,Yparam,profileList)
            if xi_arr.shape[0] == 1:
                print('error with gridding profiles...interpolating with old method...')
                # x grid in seconds, with points every 1 hour (3600 seconds)
                xi_arr = np.arange(xMinTimestamp, xMaxTimestamp, 3600)
                # y grid in meters, with points every 1/2 meter
                yi_arr = np.arange(yMin, yMax, 0.5)
                xi, yi = np.meshgrid(xi_arr, yi_arr)

                scatterX_TS = [((dt64 - unix_epoch) / one_second) for dt64 in scatterX]

                # interpolate data to grid
                zi = griddata(
                    (scatterX_TS, scatterY), scatterZ, (xi, yi), method='linear'
                )
                xiDT = xi.astype('datetime64[s]')
                # mask out any time gaps greater than 1 day
                timeGaps = np.where(np.diff(scatterX_TS) > 86400)
                if len(timeGaps[0]) > 1:
                    gaps = timeGaps[0]
                    for gap in gaps:
                        gapMask = (xi > scatterX_TS[gap]) & (xi < scatterX_TS[gap + 1])
                        zi[gapMask] = np.nan
            else:
                print('success gridding profiles...')
                xi, yi = np.meshgrid(xi_arr, yi_arr)
                ### filter out profile columns with no data where xi == 0
                zeroMask = np.where(xi_arr == 0)
                zi = np.delete(zi,zeroMask, axis=1)
                xi = np.delete(xi,zeroMask, axis=1)
                yi = np.delete(yi,zeroMask, axis=1)
                xiDT = xi.astype('datetime64[s]')
                if int(span) > 45:
                    gapThreshold = 5
                else:
                    gapThreshold = 2
                nanMask = np.where(np.diff(xiDT) > timedelta(days=gapThreshold))
                #zi[:,nanMask] = np.nan
                zi[nanMask] = np.nan

        # plot filled contours
        if zi.shape[1] > 1:
          params = {'range':'full'}
          profilePlot = plotter(xiDT, yi, zi, 'contour', colorMap, 'no', params)
          fileName = fileName_base + '_' + spanString + '_' + 'none'
          profilePlot.savefig(fileName + '_full.png', dpi=300)
          fileNameList.append(fileName + '_full.png')
          params = {'range':'local'}
          params['vmin'] = zMin
          params['vmax'] = zMax
          profilePlot = plotter(xiDT, yi, zi, 'contour', colorMap, 'no', params)
          profilePlot.savefig(fileName + '_local.png', dpi=300)
          fileNameList.append(fileName + '_local.png')
          emptySlice = 'no'
        else:
            params = {'range':'full'}
            profilePlot = plotter(0, 0, 0, 'empty', colorMap, 'Insufficient Profiles Found For Gridding', params)
            fileName = fileName_base + '_' + spanString + '_' + 'none'
            profilePlot.savefig(fileName + '_full.png', dpi=300)
            fileNameList.append(fileName + '_full.png')
            profilePlot.savefig(fileName + '_local.png', dpi=300)
            fileNameList.append(fileName + '_local.png')
            emptySlice = 'yes'
    else:
        params = {'range':'full'}
        profilePlot = plotter(0, 0, 0, 'empty', colorMap, 'No Data Available', params)
        fileName = fileName_base + '_' + spanString + '_' + 'none'
        profilePlot.savefig(fileName + '_full.png', dpi=300)
        fileNameList.append(fileName + '_full.png')
        profilePlot.savefig(fileName + '_local.png', dpi=300)
        fileNameList.append(fileName + '_local.png')
        emptySlice = 'yes'

    if 'no' in emptySlice:
        for overlay in overlays:
            if 'clim' in overlay:
                if overlayData_clim:
                    depthList = []
                    timeList = []
                    climList = []
                    for key in overlayData_clim:
                        for subKey in overlayData_clim[key]:
                            climatology = ast.literal_eval(
                                overlayData_clim[key][subKey]
                            )
                            climList.append(
                                st.mean([climatology[0], climatology[1]])
                            )
                            depthList.append(ast.literal_eval(subKey)[0])
                            timeList.append(
                                np.datetime64(
                                    "{0}-{1}-{2}".format(
                                        str(timeRef.year),
                                        str(key).zfill(2),
                                        15,
                                    ),
                                    'D',
                                )
                            )
                            # extend climatology to previous year
                            climList.append(
                                st.mean([climatology[0], climatology[1]])
                            )
                            depthList.append(ast.literal_eval(subKey)[0])
                            timeList.append(
                                np.datetime64(
                                    "{0}-{1}-{2}".format(
                                        str(timeRef.year - 1),
                                        str(key).zfill(2),
                                        15,
                                    ),
                                    'D',
                                )
                            )
                            # extend climatology to next year
                            climList.append(
                                st.mean([climatology[0], climatology[1]])
                            )
                            depthList.append(ast.literal_eval(subKey)[0])
                            timeList.append(
                                np.datetime64(
                                    "{0}-{1}-{2}".format(
                                        str(timeRef.year + 1),
                                        str(key).zfill(2),
                                        15,
                                    ),
                                    'D',
                                )
                            )

                    climTime_TS = [
                        ((dt64 - unix_epoch) / one_second) for dt64 in timeList
                    ]
                    # interpolate climatology data
                    clim_zi = griddata(
                        (climTime_TS, depthList),
                        climList,
                        (xi, yi),
                        method='linear',
                    )
                    climDiff = zi - clim_zi
                    maxLim = max(
                        abs(np.nanmin(climDiff)), abs(np.nanmax(climDiff))
                    )
                    # plot filled contours
                    climParams = {}
                    climParams['range'] = 'na'
                    climParams['norm'] = 'no'
                    climParams['vmin'] = -maxLim
                    climParams['vmax'] = maxLim
                    climPlot = plotter(xiDT, yi, climDiff, 'clim', 'cmo.balance', 'no', climParams)
                    fileName = fileName_base + '_' + spanString + '_' + 'clim'
                    climPlot.savefig(fileName + '_full.png', dpi=300)
                    fileNameList.append(fileName + '_full.png')

                    climDiffMin = np.nanmin(climDiff)
                    climDiffMax = np.nanmax(climDiff)
                    if climDiffMax < 0:
                        climDiffMax = 0
                        colorMapLocal = balanceBlue
                        divColor = 'no'
                    elif climDiffMin > 0:
                        climDiffMin = 0
                        colorMapLocal = 'cmo.amp'
                        divColor = 'no'
                    else:
                        colorMapLocal = 'cmo.balance'
                        divColor = 'yes'
                    if 'yes' in divColor:
                    #    divnorm = colors.TwoSlopeNorm(
                    #        vmin=climDiffMin, vcenter=0, vmax=climDiffMax
                    #    )
                        # plot filled contours
                        climParams = {}
                        climParams['range'] = 'na'
                        climParams['norm'] = 'yes'
                        ###climParams['norm']['divnorm'] = divnorm
                        climParams['vmin'] = climDiffMin
                        climParams['vmax'] = climDiffMax
                        climPlot = plotter(xiDT, yi, climDiff, 'clim', 'cmo.balance', 'no', climParams)
  
                    else:
                        # plot filled contours
                        climParams = {}
                        climParams['range'] = 'na'
                        climParams['norm'] = 'no'
                        climParams['vmin'] = climDiffMin
                        climParams['vmax'] = climDiffMax
                        climPlot = plotter(xiDT, yi, climDiff, 'clim', 'cmo.balance', 'no', climParams)

                    climPlot.savefig(fileName + '_local.png', dpi=300)
                    fileNameList.append(fileName + '_local.png')

                else:
                    print('climatology is empty!')
                    params = {'range':'full'}
                    profilePlot = plotter(0, 0, 0, 'empty', colorMap, 'No Climatology Data Available', params)
                    fileName = fileName_base + '_' + spanString + '_' + 'clim'
                    profilePlot.savefig(fileName + '_full.png', dpi=300)
                    fileNameList.append(fileName + '_full.png')
                    profilePlot.savefig(fileName + '_local.png', dpi=300)
                    fileNameList.append(fileName + '_local.png')

    else:
        params = {'range':'full'}
        profilePlot = plotter(0, 0, 0, 'empty', colorMap, 'No Data Available', params)
        fileName = fileName_base + '_' + spanString + '_' + 'clim'
        profilePlot.savefig(fileName + '_full.png', dpi=300)
        fileNameList.append(fileName + '_full.png')
        profilePlot.savefig(fileName + '_local.png', dpi=300)
        fileNameList.append(fileName + '_local.png')

    return fileNameList


def plotScatter(
    Yparam,
    paramData,
    plotTitle,
    yLabel,
    timeRef,
    yMin,
    yMax,
    fileName_base,
    overlayData_clim,
    overlayData_near,
    plotMarkerSize,
    span,
    spanString,
    statusDict,
    site,
):
    # Initiate fileName list
    fileNameList = []

    # Plot Overlays
    overlays = ['clim', 'near', 'time', 'none']

    # Data Ranges
    ranges = ['full', 'local']

    lineColors = [
        '#1f78b4',
        '#a6cee3',
        '#b2df8a',
        '#33a02c',
        '#ff7f00',
        '#fdbf6f',
        '#e31a1c',
        '#fb9a99',
        '#542c2c',
        '#6e409c',
    ]
    balanceBig = plt.get_cmap('cmo.balance', 512)
    balanceBlue = ListedColormap(balanceBig(np.linspace(0, 0.5, 256)))

    statusString = statusDict[site]
    statusColors = {'OPERATIONAL': 'green',
                'FAILED': 'red',
                'TROUBLESHOOTING': 'red',
                'RECOVERED': 'blue',
                'PARTIALLY_FUNCTIONAL': 'red',
                'OFFLINE': 'blue',
                'UNCABLED': 'blue',
                'DATA_QUALITY': 'red',
                'NOT_DEPLOYED': 'blue'
                }


    def setPlot():

        plt.close('all')
        plt.rcParams["font.family"] = "serif"

        fig, ax = plt.subplots()
        fig.set_size_inches(5, 1.75)
        fig.patch.set_facecolor('white')
        plt.title(plotTitle, fontsize=4, loc='left')
        plt.title(statusString, fontsize=4, fontweight=0, color=statusColors[statusString], loc='right', style='italic' )
        plt.ylabel(yLabel, fontsize=4)
        ax.tick_params(direction='out', length=2, width=0.5, labelsize=4)
        ax.ticklabel_format(useOffset=False)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats = [
            '%y',  # ticks are mostly years
            '%b',  # ticks are mostly months
            '%m/%d',  # ticks are mostly days
            '%H h',  # hrs
            '%H:%M',  # min
            '%S.%f',
        ]  # secs
        formatter.zero_formats = [
            '',  # ticks are mostly years, no need for zero_format
            '%b-%Y',  # ticks are mostly months, mark month/year
            '%m/%d',  # ticks are mostly days, mark month/year
            '%m/%d',  # ticks are mostly hours, mark month and day
            '%H',  # ticks are montly mins, mark hour
            '%M',
        ]  # ticks are mostly seconds, mark minute

        formatter.offset_formats = [
            '',
            '',
            '',
            '',
            '',
            '',
        ]

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(False)
        return (fig, ax)

    print('plotting scatter for timeSpan: ', span)

    if 'deploy' in spanString:
        deployHistory = loadDeploymentHistory(site)
        deployTimes = listDeployTimes(deployHistory[site])

        timeRef_deploy = deployTimes[0]
        startDate = timeRef_deploy - timedelta(days=15)
        endDate = timeRef_deploy + timedelta(days=15)
        xMin = startDate - timedelta(days=15 * 0.002)
        xMax = endDate + timedelta(days=15 * 0.002)
    else:
        endDate = timeRef
        startDate = timeRef - timedelta(days=int(span))
        xMin = startDate - timedelta(days=int(span) * 0.002)
        xMax = endDate + timedelta(days=int(span) * 0.002)

    baseDS = paramData.sel(time=slice(startDate, endDate))
    scatterX = baseDS.time.values
    scatterY = np.array([])
    if len(scatterX) > 0:
        scatterY = baseDS.values
    if ('small' in plotMarkerSize) & (len(scatterX) < 1000):
        plotMarkerSize = 'medium'
    fig, ax = setPlot()
    emptySlice = 'no'
    if 'large' in plotMarkerSize:
        plt.plot(scatterX, scatterY, '.', color=lineColors[0], markersize=2)
    elif 'medium' in plotMarkerSize:
        plt.plot(scatterX, scatterY, '.', color=lineColors[0], markersize=0.75)
    elif 'small' in plotMarkerSize:
        plt.plot(scatterX, scatterY, ',', color=lineColors[0])
    plt.xlim(xMin, xMax)
    # ylim_current = plt.gca().get_ylim()
    if scatterX.size == 0:
        print('slice is empty!')
        plt.annotate(
            'No Data Available', xy=(0.3, 0.5), xycoords='axes fraction'
        )
        emptySlice = 'yes'
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    for axis in ['top','bottom','left','right']:
        cax.spines[axis].set_linewidth(0)
    cax.set_xticks([])
    cax.set_yticks([])
    fileName = fileName_base + '_' + spanString + '_' + 'none'
    fig.savefig(fileName + '_full.png', dpi=300)
    fileNameList.append(fileName + '_full.png')
    ax.set_ylim(yMin, yMax)
    fig.savefig(fileName + '_local.png', dpi=300)
    fileNameList.append(fileName + '_local.png')

    for overlay in overlays:
        if 'time' in overlay:
            fig, ax = setPlot()
            
            print('adding time machine plot')
            timeMachineList = []
            if 'deploy' in spanString:
                xMinTime = parser.parse(str(deployTimes[0].year) + '-05-01')
                xMaxTime = parser.parse(str(deployTimes[0].year) + '-10-01')
                plt.xlim(xMinTime, xMaxTime)
                yearRef = deployTimes[0].year
                for time in deployTimes:
                    start = time - timedelta(days=15)
                    end = time + timedelta(days=15)
                    timeMachineList.append([time,start,end]) 
            else:
                plt.xlim(xMin, xMax)
                yearRef = timeRef.year
                start = timeRef - timedelta(days=int(span))
                timeMachineList.append([timeRef,start,timeRef])
                startYear = pd.to_datetime(paramData['time'].values.min()).year
                numYears = timeRef.year - startYear
                years = np.arange(1,numYears+1,1)
                for year in years:
                    time = timeRef - timedelta(days=int(year*365))
                    start = time - timedelta(days=int(span))
                    end = time
                    timeMachineList.append([time,start,end])
            
            for timeTrace in timeMachineList:
                yearDiff = int(yearRef) - int(timeTrace[0].year)
                timeDS = paramData.sel(time=slice(timeTrace[1],timeTrace[2]))
                if timeDS.time.size !=0:
                    minYear = pd.to_datetime(timeDS['time'].values.min()).year
                    maxYear = pd.to_datetime(timeDS['time'].values.max()).year
                    if minYear != maxYear:
                        legendString = f'{minYear} - {maxYear}'
                    else:
                        legendString = f'{maxYear}'
                    timeDS['plotTime'] = timeDS.time + np.timedelta64(timedelta(days=365 * yearDiff))
                    timeX = timeDS.plotTime.values
                    timeY = np.array([])
                    if len(timeX) > 0:
                        timeY = timeDS.values
                    c = lineColors[yearDiff]
                    if 'large' in plotMarkerSize:
                        plt.plot(timeX, timeY,'.',markersize=2,c=c,label='%s' % legendString,)
                    elif 'medium' in plotMarkerSize:
                        plt.plot(timeX,timeY,'.',markersize=0.75,c=c,label='%s' % legendString,)
                    elif 'small' in plotMarkerSize:
                        plt.plot(timeX,timeY,',',c=c,label='%s' % legendString,)
                del timeDS
                gc.collect()

            # generating custom legend
            handles, labels = ax.get_legend_handles_labels()
            patches = []
            for handle, label in zip(handles, labels):
                patches.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=handle.get_color(),
                        marker='o',
                        markersize=1,
                        linewidth=0,
                        label=label,
                    )
                )

            legend = ax.legend(handles=patches, loc="upper right", fontsize=3)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            for axis in ['top','bottom','left','right']:
                cax.spines[axis].set_linewidth(0)
            cax.set_xticks([])
            cax.set_yticks([])
            fileName = fileName_base + '_' + spanString + '_' + overlay
            fig.savefig(fileName + '_full.png', dpi=300)
            fileNameList.append(fileName + '_full.png')
            ax.set_ylim(yMin, yMax)
            fig.savefig(fileName + '_local.png', dpi=300)
            fileNameList.append(fileName + '_local.png')

        if 'clim' in overlay:
            # add climatology trace
            print('adding climatology trace to plot')
            if 'no' in emptySlice:
                fig, ax = setPlot()
                plt.xlim(xMin, xMax)
                if not overlayData_clim.empty:
                    if 'large' in plotMarkerSize:
                        plt.plot(
                            scatterX,
                            scatterY,
                            '.',
                            color=lineColors[0],
                            markersize=2,
                        )
                    elif 'medium' in plotMarkerSize:
                        plt.plot(
                            scatterX,
                            scatterY,
                            '.',
                            color=lineColors[0],
                            markersize=0.75,
                        )
                    elif 'small' in plotMarkerSize:
                        plt.plot(scatterX, scatterY, ',', color=lineColors[0])

                    plt.fill_between(
                        overlayData_clim.index,
                        overlayData_clim.climMinus3std,
                        overlayData_clim.climPlus3std,
                        alpha=0.2,
                    )
                    plt.plot(
                        overlayData_clim.climData,
                        '-.',
                        color='r',
                        alpha=0.4,
                        linewidth=0.25,
                    )
                else:
                    print('Climatology is empty!')
                    plt.annotate(
                        'No Climatology Data Available',
                        xy=(0.3, 0.5),
                        xycoords='axes fraction',
                    )
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2%", pad=0.05)
                for axis in ['top','bottom','left','right']:
                    cax.spines[axis].set_linewidth(0)
                cax.set_xticks([])
                cax.set_yticks([])
                fileName = fileName_base + '_' + spanString + '_' + 'clim'
                fig.savefig(fileName + '_full.png', dpi=300)
                fileNameList.append(fileName + '_full.png')
                ax.set_ylim(yMin, yMax)
                fig.savefig(fileName + '_local.png', dpi=300)
                fileNameList.append(fileName + '_local.png')

        if 'near' in overlay:
            # add nearest neighbor data traces
            print('adding nearest neighbor data to plot')
    return fileNameList
