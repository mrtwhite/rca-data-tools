# -*- coding: utf-8 -*-
"""dashboard.py

This module contains code for creating pngs to feed into the QAQC dashboard.

"""

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

    renameMap = {
                 'sea_water_temperature':'seawater_temperature',
                 'sea_water_practical_salinity':'practical_salinity',
                 'sea_water_pressure':'seawater_pressure',
                 'sea_water_density':'density',
                 'ph_seawater':'seawater_ph',
                 }

    if param in renameMap:
        param = renameMap[param]

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
    zMin_local,
    zMax_local,
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
    ranges = ['full', 'standard', 'local']

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
            if 'full' in params['range']:
                graph = ax.contourf(Xx, Yy, Zz, 50, cmap=colorBar)
            else:
                colorRange = params['vmax'] - params['vmin']
                cbarticks = np.arange(params['vmin'],params['vmax'],colorRange/50)
                graph = ax.contourf(Xx, Yy, Zz, cbarticks, cmap=colorBar)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            cbar = plt.colorbar(graph, cax=cax)
            if 'standard' in params['range']:
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
          if 'deploy' in spanString:
              plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
          fileName = fileName_base + '_' + spanString + '_' + 'none'
          profilePlot.savefig(fileName + '_full.png', dpi=300)
          fileNameList.append(fileName + '_full.png')
          params = {'range':'standard'}
          params['vmin'] = zMin
          params['vmax'] = zMax
          profilePlot = plotter(xiDT, yi, zi, 'contour', colorMap, 'no', params)
          if 'deploy' in spanString:
              plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
          profilePlot.savefig(fileName + '_standard.png', dpi=300)
          fileNameList.append(fileName + '_standard.png')
          params = {'range':'local'}
          params['vmin'] = zMin_local
          params['vmax'] = zMax_local
          profilePlot = plotter(xiDT, yi, zi, 'contour', colorMap, 'no', params)
          if 'deploy' in spanString:
              plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
          profilePlot.savefig(fileName + '_local.png', dpi=300)
          fileNameList.append(fileName + '_local.png')
          emptySlice = 'no'
        else:
            params = {'range':'full'}
            profilePlot = plotter(0, 0, 0, 'empty', colorMap, 'Insufficient Profiles Found For Gridding', params)
            fileName = fileName_base + '_' + spanString + '_' + 'none'
            profilePlot.savefig(fileName + '_full.png', dpi=300)
            fileNameList.append(fileName + '_full.png')
            profilePlot.savefig(fileName + '_standard.png', dpi=300)
            fileNameList.append(fileName + '_standard.png')
            profilePlot.savefig(fileName + '_local.png', dpi=300)
            fileNameList.append(fileName + '_local.png')
            emptySlice = 'yes'
    else:
        params = {'range':'full'}
        profilePlot = plotter(0, 0, 0, 'empty', colorMap, 'No Data Available', params)
        fileName = fileName_base + '_' + spanString + '_' + 'none'
        profilePlot.savefig(fileName + '_full.png', dpi=300)
        fileNameList.append(fileName + '_full.png')
        profilePlot.savefig(fileName + '_standard.png', dpi=300)
        fileNameList.append(fileName + '_standard.png')
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
                    if 'deploy' in spanString:
                        plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
                    fileName = fileName_base + '_' + spanString + '_' + 'clim'
                    climPlot.savefig(fileName + '_full.png', dpi=300)
                    fileNameList.append(fileName + '_full.png')

                    climDiffMin = np.nanmin(climDiff)
                    climDiffMax = np.nanmax(climDiff)
                    if climDiffMax < 0:
                        climDiffMax = 0
                        colorMapStandard = balanceBlue
                        divColor = 'no'
                    elif climDiffMin > 0:
                        climDiffMin = 0
                        colorMapStandard = 'cmo.amp'
                        divColor = 'no'
                    else:
                        colorMapStandard = 'cmo.balance'
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
                        climPlot = plotter(xiDT, yi, climDiff, 'clim', colorMapStandard, 'no', climParams)
  
                    else:
                        # plot filled contours
                        climParams = {}
                        climParams['range'] = 'na'
                        climParams['norm'] = 'no'
                        climParams['vmin'] = climDiffMin
                        climParams['vmax'] = climDiffMax
                        climPlot = plotter(xiDT, yi, climDiff, 'clim', colorMapStandard, 'no', climParams)
       
                    if 'deploy' in spanString:
                        plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
                    climPlot.savefig(fileName + '_standard.png', dpi=300)
                    fileNameList.append(fileName + '_standard.png')
                    climPlot.savefig(fileName + '_local.png', dpi=300)
                    fileNameList.append(fileName + '_local.png')

                else:
                    print('climatology is empty!')
                    params = {'range':'full'}
                    profilePlot = plotter(0, 0, 0, 'empty', colorMap, 'No Climatology Data Available', params)
                    fileName = fileName_base + '_' + spanString + '_' + 'clim'
                    profilePlot.savefig(fileName + '_full.png', dpi=300)
                    fileNameList.append(fileName + '_full.png')
                    profilePlot.savefig(fileName + '_standard.png', dpi=300)
                    fileNameList.append(fileName + '_standard.png')
                    profilePlot.savefig(fileName + '_local.png', dpi=300)
                    fileNameList.append(fileName + '_local.png')


    else:
        params = {'range':'full'}
        profilePlot = plotter(0, 0, 0, 'empty', colorMap, 'No Data Available', params)
        fileName = fileName_base + '_' + spanString + '_' + 'clim'
        profilePlot.savefig(fileName + '_full.png', dpi=300)
        fileNameList.append(fileName + '_full.png')
        profilePlot.savefig(fileName + '_standard.png', dpi=300)
        fileNameList.append(fileName + '_standard.png')
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
    yMin_local,
    yMax_local,
    fileName_base,
    overlayData_clim,
    overlayData_flag,
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
    overlays = ['clim', 'flag', 'near', 'time', 'none']

    # Data Ranges
    ranges = ['full', 'standard', 'local']

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
    if 'deploy' in spanString:
        plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
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
    fig.savefig(fileName + '_standard.png', dpi=300)
    fileNameList.append(fileName + '_standard.png')
    ax.set_ylim(yMin_local, yMax_local)
    fig.savefig(fileName + '_local.png', dpi=300)
    fileNameList.append(fileName + '_local.png')


    for overlay in overlays:
        if 'time' in overlay:
            fig, ax = setPlot()
            
            print('adding time machine plot')
            timeMachineList = []
            if 'deploy' in spanString:
                xMinTime = parser.parse(str(deployTimes[0].year) + '-06-15')
                xMaxTime = parser.parse(str(deployTimes[0].year) + '-09-15')
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
                    if 'deploy' in spanString:
                        deployTime_plot = timeTrace[0] + np.timedelta64(timedelta(days=365 * yearDiff))
                        plt.axvline(deployTime_plot,linewidth=1,color=c,linestyle='-.')
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
            fig.savefig(fileName + '_standard.png', dpi=300)
            fileNameList.append(fileName + '_standard.png')
            ax.set_ylim(yMin_local, yMax_local)
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
                if 'deploy' in spanString:
                    plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
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
                fig.savefig(fileName + '_standard.png', dpi=300)
                fileNameList.append(fileName + '_standard.png')
                ax.set_ylim(yMin_local, yMax_local)
                fig.savefig(fileName + '_local.png', dpi=300)
                fileNameList.append(fileName + '_local.png')

        if 'near' in overlay:
            # add nearest neighbor data traces
            print('adding nearest neighbor data to plot')

        if 'flag' in overlay:
            # highlight flagged data points
            print('adding flagged data overlay to plot')
            if 'no' in emptySlice:
                fig, ax = setPlot()
                plt.xlim(xMin, xMax)
                legendString = 'all data'
                if 'large' in plotMarkerSize:
                    flagMarker = 3
                    plt.plot(
                        scatterX, 
                        scatterY,
                        '.',
                        color=lineColors[0],
                        markersize=2,
                        label='%s' % legendString,
                    )
                elif 'medium' in plotMarkerSize:
                    flagMarker = 1.5
                    plt.plot(
                        scatterX,
                        scatterY,
                        '.',
                        color=lineColors[0],
                        markersize=0.75,
                        label='%s' % legendString,
                    )
                elif 'small' in plotMarkerSize:
                    flagMarker = 0.25
                    plt.plot(scatterX, scatterY, ',', color=lineColors[0], label='%s' % legendString,)
                # slice overlayData_flag
                qcDS = overlayData_flag.sel(time=slice(startDate, endDate))
                # retrieve flags
                qcDS = retrieve_qc(qcDS)
                flags = {
                    ##'qartod_grossRange':{'symbol':'+', 'param':'_qartod_gr_flag'},
                    ##'qartod_climatology':{'symbol':'x','param':'_qartod_cl_flag'},
                    'qartod_summary':{'symbol':'1','param':'_qartod_results'},
                    'qc':{'symbol':'s','param':'_qc_summary_flag'},
                }
                for flagType in flags.keys():
                    flagString = Yparam + flags[flagType]['param']
                    print(flagString)
                    if flagString in qcDS:
                        print('paramters found for ',flagString)
                        flagStatus = {'fail':{'value':4,'color':'r'}, 'suspect':{'value':3,'color':'y'}}
                        for level in flagStatus.keys():
                            flaggedDS = qcDS.where(qcDS[flagString] == flagStatus[level]['value'], drop=True)
                            flag_X = flaggedDS.time.values
                            if len(flag_X) > 0:
                                n = len(flag_X)
                                legendString = f'{flagType} {level}: {n} points'
                                flag_Y = flaggedDS[Yparam].values
                                plt.plot(
                                    flag_X,
                                    flag_Y,
                            	    flags[flagType]['symbol'],
                            	    color=flagStatus[level]['color'],
                            	    markersize=flagMarker,
                            	    label='%s' % legendString,   
                            	    )
                            else:
                                legendString = f'{flagType} {level}: no points flagged'
                                plt.plot([0],[0],color='w',markersize=0,label='%s' % legendString,)
                    else:
                        print('no paramters found for ',flagString)
                        legendString = f'no {flagType} flags found'
                        plt.plot(scatterX,scatterY,alpha=0,markersize=0,label='%s' % legendString,)

                # generating custom legend 
                handles, labels = ax.get_legend_handles_labels()
                patches = []
                for handle, label in zip(handles, labels):
                    patches.append(
                        mlines.Line2D(
                            [],  
                            [],
                            color=handle.get_color(),
                            marker=handle.get_marker(),
                            markersize=1,
                            linewidth=0,  
                            label=label,
                        )
                    )
                  
                legend = ax.legend(handles=patches, loc="upper right", fontsize=3)


                if 'deploy' in spanString:
                    plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2%", pad=0.05)
                for axis in ['top','bottom','left','right']:
                    cax.spines[axis].set_linewidth(0)
                cax.set_xticks([])
                cax.set_yticks([])
                fileName = fileName_base + '_' + spanString + '_' + 'flag'
                fig.savefig(fileName + '_full.png', dpi=300)
                fileNameList.append(fileName + '_full.png')
                ax.set_ylim(yMin, yMax)
                fig.savefig(fileName + '_standard.png', dpi=300)
                fileNameList.append(fileName + '_standard.png')
                ax.set_ylim(yMin_local, yMax_local)
                fig.savefig(fileName + '_local.png', dpi=300)
                fileNameList.append(fileName + '_local.png')

    return fileNameList



def retrieve_qc(ds):
    """
    Extract the QC test results from the different variables in the data set,
    and create a new variable with the QC test results set to match the logic
    used in QARTOD testing. Instead of setting the results to an integer
    representation of a bitmask, use the pass = 1, not_evaluated = 2,
    suspect_or_of_high_interest = 3, fail = 4 and missing = 9 flag values from
    QARTOD.
    The QC portion of this code was copied from the ooi-data-explorations parse_qc function, 
    which was was inspired by an example notebook developed by the OOI Data
    Team for the 2018 Data Workshops. The original example, by Friedrich Knuth,
    and additional information on the original OOI QC algorithms can be found
    at:
    https://oceanobservatories.org/knowledgebase/interpreting-qc-variables-and-results/
    :param ds: dataset with *_qc_executed and *_qc_results variables
               as well as qartod_executed variables if available
    :return ds: dataset with the *_qc_executed and *_qc_results variables
        reworked to create a new *_qc_summary variable with the results
        of the QC checks decoded into a QARTOD style flag value, as well as 
        extracted qartod variables (gross range and climatology).  Code will need to be
        adapted as more tests are added...
    """
    # create a list of the variables that have had QC tests applied
    variables = [x.split('_qc_results')[0] for x in ds.variables if 'qc_results' in x]

    # for each variable with qc tests applied
    for var in variables:
        # set the qc_results and qc_executed variable names and the new qc_flags variable name
        qc_result = var + '_qc_results'
        qc_executed = var + '_qc_executed'
        qc_summary = var + '_qc_summary_flag'

        # create the initial qc_flags array
        flags = np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 0]), (len(ds.time), 1))
        # the list of tests run, and their bit positions are:
        #    0: dataqc_globalrangetest
        #    1: dataqc_localrangetest
        #    2: dataqc_spiketest
        #    3: dataqc_polytrendtest
        #    4: dataqc_stuckvaluetest
        #    5: dataqc_gradienttest
        #    6: undefined
        #    7: dataqc_propagateflags

        # use the qc_executed variable to determine which tests were run, and set up a bit mask to pull out the results
        executed = np.bitwise_or.reduce(ds[qc_executed].values.astype('uint8'))
        executed_bits = np.unpackbits(executed.astype('uint8'))

        # for each test executed, reset the qc_flags for pass == 1, suspect == 3, or fail == 4
        for index, value in enumerate(executed_bits[::-1]):
            if value:
                if index in [2, 3, 4, 5, 6, 7]:
                    # mark these tests as missing since they are problematic
                    flag = 9
                else:
                    # only mark the global range test as fail, all the other tests are problematic
                    flag = 4
                mask = 2 ** index
                m = (ds[qc_result].values.astype('uint8') & mask) > 0
                flags[m, index] = 1   # True == pass
                flags[~m, index] = flag  # False == suspect/fail

        # add the qc_flags to the dataset, rolling up the results into a single value
        ds[qc_summary] = ('time', flags.max(axis=1, initial=1).astype(np.int32))

    # create a list of the variables that have had QARTOD tests applied
    variables = [x.split('_qartod_executed')[0] for x in ds.variables if 'qartod_executed' in x]

    # for each variable with qc tests applied
    for var in variables:
        qartodString = var + '_qartod_executed'
        flagNameBase = var + '_qartod_'
        testOrder = ds[qartodString][0].tests_executed.strip("'").replace(" ","").split(',')
        for i in range(0, len(testOrder)):
            flagString = testOrder[i]
            flagIndex = testOrder.index(flagString)
            flagName = flagNameBase + flagString
            ds[flagName] = [int(i[flagIndex]) for i in ds[qartodString].values.tolist()]

    return ds


