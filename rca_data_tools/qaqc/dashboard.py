# -*- coding: utf-8 -*-
import ast
from datetime import datetime, timedelta
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


def loadData(site, sites_dict):
    fs = s3fs.S3FileSystem(anon=True)
    zarrDir = 'ooi-data/' + sites_dict[site]['zarrFile']
    zarr_store = fs.get_mapper(zarrDir)
    # TODO: only request parameters listed in sites_dict[site][dataParameters]?
    # requestParams = sites_dict[site]['dataParameters'].strip('"').split(',')
    ds = xr.open_zarr(zarr_store, consolidated=True)

    return ds


def plotProfilesGrid(
    Yparam,
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
        '#fb9a99',
        '#e31a1c',
        '#fdbf6f',
        '#ff7f00',
    ]
    balanceBig = plt.get_cmap('cmo.balance', 512)
    balanceBlue = ListedColormap(balanceBig(np.linspace(0, 0.5, 256)))

    def setPlot():

        plt.close('all')
        plt.rcParams["font.family"] = "serif"

        fig, ax = plt.subplots()
        fig.set_size_inches(5.65, 1.75)
        fig.patch.set_facecolor('white')
        plt.title(plotTitle, fontsize=4, loc='left')
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
        return (fig, ax)

    endDate = timeRef

    print('plotting timeSpan: ', span)
    startDate = timeRef - timedelta(days=int(span))
    xMin = startDate - timedelta(days=int(span) * 0.002)
    xMax = endDate + timedelta(days=int(span) * 0.002)
    baseDS = paramData.sel(time=slice(startDate, endDate))
    scatterX = baseDS.time.values
    scatterY = np.array([])
    scatterZ = np.array([])
    if len(scatterX) > 0:
        scatterY = baseDS.seawater_pressure.values
        scatterZ = baseDS[Yparam].values
    fig, ax = setPlot()

    if scatterX.size != 0:
        # create interpolation grid
        xMinTimestamp = xMin.timestamp()
        xMaxTimestamp = xMax.timestamp()
        xi = np.arange(xMinTimestamp, xMaxTimestamp, 3600)
        yi = np.arange(yMin, yMax, 0.5)
        xi, yi = np.meshgrid(xi, yi)

        unix_epoch = np.datetime64(0, 's')
        one_second = np.timedelta64(1, 's')
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

        # plot filled contours
        profilePlot = plt.contourf(xiDT, yi, zi, 50, cmap=colorMap)
        emptySlice = 'no'
    else:
        print('slice is empty!')
        profilePlot = plt.scatter(
            scatterX, scatterY, c=scatterZ, marker='.', cmap=colorMap
        )
        plt.annotate(
            'No data available', xy=(0.3, 0.5), xycoords='axes fraction'
        )
        emptySlice = 'yes'

    plt.xlim(xMin, xMax)
    cbar = fig.colorbar(profilePlot, ax=ax)
    cbar.update_ticks()
    cbar.formatter.set_useOffset(False)
    cbar.ax.set_ylabel(zLabel, fontsize=4)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=4)

    fileName = fileName_base + '_' + spanString + '_' + 'none'
    fig.savefig(fileName + '_full.png', dpi=300)
    fileNameList.append(fileName + '_full.png')
    cbar.remove()
    plt.clim(zMin, zMax)
    m = ScalarMappable(cmap=profilePlot.get_cmap())
    m.set_array(profilePlot.get_array())
    m.set_clim(profilePlot.get_clim())
    cbar = fig.colorbar(m, ax=ax)
    cbar.update_ticks()
    cbar.formatter.set_useOffset(False)
    cbar.ax.set_ylabel(zLabel, fontsize=4)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=4)
    fig.savefig(fileName + '_local.png', dpi=300)
    fileNameList.append(fileName + '_local.png')

    if 'no' in emptySlice:
        for overlay in overlays:
            if 'clim' in overlay:
                fig, ax = setPlot()
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
                    profilePlot = plt.contourf(
                        xiDT,
                        yi,
                        climDiff,
                        50,
                        cmap='cmo.balance',
                        vmin=-maxLim,
                        vmax=maxLim,
                    )
                    plt.clim(-maxLim, maxLim)
                    m = ScalarMappable(cmap=profilePlot.get_cmap())
                    m.set_array(profilePlot.get_array())
                    m.set_clim(profilePlot.get_clim())
                    cbar = fig.colorbar(m, ax=ax)
                    cbar.update_ticks()
                    cbar.formatter.set_useOffset(False)
                    cbar.ax.set_ylabel(zLabel, fontsize=4)
                    cbar.ax.tick_params(length=2, width=0.5, labelsize=4)
                    plt.xlim(xMin, xMax)

                    fileName = fileName_base + '_' + spanString + '_' + 'clim'
                    fig.savefig(fileName + '_full.png', dpi=300)
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

                    fig, ax = setPlot()
                    if 'yes' in divColor:
                        divnorm = colors.TwoSlopeNorm(
                            vmin=climDiffMin, vcenter=0, vmax=climDiffMax
                        )
                        profilePlot = plt.contourf(
                            xiDT,
                            yi,
                            climDiff,
                            50,
                            cmap=colorMapLocal,
                            vmin=climDiffMin,
                            vmax=climDiffMax,
                            norm=divnorm,
                        )
                        cbar = fig.colorbar(profilePlot, ax=ax)
                    else:
                        profilePlot = plt.contourf(
                            xiDT,
                            yi,
                            climDiff,
                            50,
                            cmap=colorMapLocal,
                            vmin=climDiffMin,
                            vmax=climDiffMax,
                        )

                        plt.clim(climDiffMin, climDiffMax)
                        m = ScalarMappable(cmap=profilePlot.get_cmap())
                        m.set_array(profilePlot.get_array())
                        m.set_clim(profilePlot.get_clim())
                        cbar = fig.colorbar(m, ax=ax)

                    cbar.update_ticks()
                    cbar.formatter.set_useOffset(False)
                    cbar.ax.set_ylabel(zLabel, fontsize=4)
                    cbar.ax.tick_params(length=2, width=0.5, labelsize=4)
                    plt.xlim(xMin, xMax)

                    fig.savefig(fileName + '_local.png', dpi=300)
                    fileNameList.append(fileName + '_local.png')

                else:
                    print('climatology is empty!')
                    plt.annotate(
                        'No climatology data available',
                        xy=(0.3, 0.5),
                        xycoords='axes fraction',
                    )

                    fileName = fileName_base + '_' + spanString + '_' + 'clim'
                    fig.savefig(fileName + '_full.png', dpi=300)
                    fileNameList.append(fileName + '_full.png')
                    fig.savefig(fileName + '_local.png', dpi=300)
                    fileNameList.append(fileName + '_local.png')

    else:
        fig, ax = setPlot()
        profilePlot = plt.scatter(
            scatterX, scatterY, c=scatterZ, marker='.', cmap='cmo.balance'
        )
        plt.annotate(
            'No data available', xy=(0.3, 0.5), xycoords='axes fraction'
        )
        plt.xlim(xMin, xMax)
        cbar = fig.colorbar(profilePlot, ax=ax)
        cbar.update_ticks()
        cbar.formatter.set_useOffset(False)
        cbar.ax.set_ylabel(zLabel, fontsize=4)
        cbar.ax.tick_params(length=2, width=0.5, labelsize=4)
        fileName = fileName_base + '_' + spanString + '_' + 'clim'
        fig.savefig(fileName + '_full.png', dpi=300)
        fileNameList.append(fileName + '_full.png')
        fig.savefig(fileName + '_local.png', dpi=300)
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
        '#fb9a99',
        '#e31a1c',
        '#fdbf6f',
        '#ff7f00',
    ]
    balanceBig = plt.get_cmap('cmo.balance', 512)
    balanceBlue = ListedColormap(balanceBig(np.linspace(0, 0.5, 256)))

    def setPlot():

        plt.close('all')
        plt.rcParams["font.family"] = "serif"

        fig, ax = plt.subplots()
        fig.set_size_inches(4.5, 1.75)
        fig.patch.set_facecolor('white')
        plt.title(plotTitle, fontsize=4, loc='left')
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

    endDate = timeRef
    print('plotting timeSpan: ', span)
    startDate = timeRef - timedelta(days=int(span))
    xMin = startDate - timedelta(days=int(span) * 0.002)
    xMax = endDate + timedelta(days=int(span) * 0.002)
    baseDS = paramData.sel(time=slice(startDate, endDate))
    scatterX = baseDS.time.values
    scatterY = np.array([])
    if len(scatterX) > 0:
        scatterY = baseDS.values
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
            'No data available', xy=(0.3, 0.5), xycoords='axes fraction'
        )
        emptySlice = 'yes'
    fileName = fileName_base + '_' + spanString + '_' + 'none'
    fig.savefig(fileName + '_full.png', dpi=300)
    fileNameList.append(fileName + '_full.png')
    plt.ylim(yMin, yMax)
    fig.savefig(fileName + '_local.png', dpi=300)
    fileNameList.append(fileName + '_local.png')

    for overlay in overlays:
        if 'time' in overlay:
            fig, ax = setPlot()
            plt.xlim(xMin, xMax)
            # plot previous 6 years of data slices for timespan
            print('adding time machine plot')
            # TODO: make this a smarter iterator about how many years of data exist...
            numYears = 6
            traces = []
            for z in range(0, numYears):
                timeRef_year = timeRef - timedelta(days=z * 365)
                time_startDate = timeRef_year - timedelta(days=int(span))
                time_endDate = timeRef_year
                timeDS = paramData.sel(
                    time=slice(time_startDate, time_endDate)
                )
                timeDS['plotTime'] = timeDS.time + np.timedelta64(
                    timedelta(days=365 * z)
                )
                timeX = timeDS.plotTime.values
                timeY = np.array([])
                if len(timeX) > 0:
                    timeY = timeDS.values
                c = lineColors[z]
                if 'large' in plotMarkerSize:
                    plt.plot(
                        timeX,
                        timeY,
                        '.',
                        markersize=2,
                        c=c,
                        label='%s' % str(timeRef_year.year),
                    )
                elif 'medium' in plotMarkerSize:
                    plt.plot(
                        timeX,
                        timeY,
                        '.',
                        markersize=0.75,
                        c=c,
                        label='%s' % str(timeRef_year.year),
                    )
                elif 'small' in plotMarkerSize:
                    plt.plot(
                        timeX,
                        timeY,
                        ',',
                        c=c,
                        label='%s' % str(timeRef_year.year),
                    )
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
            fileName = fileName_base + '_' + spanString + '_' + overlay
            fig.savefig(fileName + '_full.png', dpi=300)
            fileNameList.append(fileName + '_full.png')
            plt.ylim(yMin, yMax)
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
                        'No climatology data available',
                        xy=(0.3, 0.5),
                        xycoords='axes fraction',
                    )

                fileName = fileName_base + '_' + spanString + '_' + 'clim'
                fig.savefig(fileName + '_full.png', dpi=300)
                fileNameList.append(fileName + '_full.png')
                plt.ylim(yMin, yMax)
                fig.savefig(fileName + '_local.png', dpi=300)
                fileNameList.append(fileName + '_local.png')

        if 'near' in overlay:
            # add nearest neighbor data traces
            print('adding nearest neighbor data to plot')
    return fileNameList
