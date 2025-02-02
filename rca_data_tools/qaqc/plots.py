# -*- coding: utf-8 -*-
"""plots.py

This module contains code for plot creations from various instruments.

"""

import argparse
from ast import literal_eval
import concurrent.futures
from datetime import datetime
from dateutil import parser
import gc
import pandas as pd
from pathlib import Path
import xarray as xr

from rca_data_tools.qaqc import dashboard
from rca_data_tools.qaqc import decimate
from rca_data_tools.qaqc.utils import coerce_qartod_executed_to_int
from rca_data_tools.qaqc.flow import S3_BUCKET
from rca_data_tools.qaqc.utils import select_logger

HERE = Path(__file__).parent.absolute()
PARAMS_DIR = HERE.joinpath('params')
PLOT_DIR = Path('QAQC_plots')

selection_mapping = {
    'ctd-profiler': 'CTD-PROFILER',
    'ctd-fixed': 'CTD-FIXED',
    'ctd-fixed-xo2': 'CTD-FIXED-XO2',
    'flr-profiler': 'FLR-PROFILER',
    'flr-fixed': 'FLR-FIXED',
    'nut-profiler': 'NUT-PROFILER',
    'nut-fixed': 'NUT-FIXED',
    'par-profiler': 'PAR-PROFILER',
    'pco2-profiler': 'PCO2-PROFILER',
    'pco2-fixed': 'PCO2-FIXED',
    'ph-profiler': 'PH-PROFILER',
    'ph-fixed': 'PH-FIXED',
    'spkir-profiler': 'SPKIR-PROFILER',
    'velpt-profiler': 'VELPT-PROFILER',
}
span_dict = {
    '1': 'day',
    '7': 'week',
    '30': 'month',
    '365': 'year',
    '0': 'deploy',
}

# create dictionary of sites key for filePrefix, nearestNeighbors
sites_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('sitesDictionary.csv'))
    .set_index('refDes')
    .T.to_dict('series')
)

# create dictionary of parameter vs variable Name
variable_dict = pd.read_csv(
    PARAMS_DIR.joinpath('variableMap.csv'), index_col=0, squeeze=True
).to_dict()

# create dictionary of instrumet key for plot parameters
instrument_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('plotParameters.csv'))
    .set_index('instrument')
    .T.to_dict('series')
)

# create dictionary of variable parameters for plotting
variable_paramDict = (
    pd.read_csv(PARAMS_DIR.joinpath('variableParameters.csv'))
    .set_index('variable')
    .T.to_dict('series')
)

# create dictionary of multi-parameter instrumet variables
multiParameter_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('multiParameters.csv'))
    .set_index('instrument')
    .T.to_dict('series')
)

# create dictionary of local parameter ranges for each site
localRange_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('localRanges.csv'))
    .set_index('refDes')
    .T.to_dict('series')
)

# load status dictionary
statusDict = dashboard.loadStatus()

plotDir = str(PLOT_DIR) + '/'


def extractMulti(ds, inst, multi_dict, fileParams):
    multiParam = multi_dict[inst]['parameter']
    subParams = multi_dict[inst]['subParameters'].strip('"').split(',')
    for i in range(0, len(subParams)):
        newParam = multiParam + '_' + subParams[i]
        ds[newParam] = ds[multiParam][:, i]
        fileParams.append(newParam)
    return ds, fileParams


def map_concurrency(
    func, iterator, func_args=(), func_kwargs={}, max_workers=10
):
    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {
            executor.submit(func, i, *func_args, **func_kwargs): i
            for i in iterator
        }
        for future in concurrent.futures.as_completed(future_to_url):
            data = future.result()
            results.append(data)
    return results


def run_dashboard_creation(
    site,
    paramList,
    timeRef,
    plotInstrument,
    span,
    decimationThreshold,
):
    from prefect import get_run_logger
    try:
        logger = get_run_logger()
    except:
        print('Could not start prefect logger...running local log')
        from loguru import logger

    if isinstance(timeRef, str):
        timeRef = parser.parse(timeRef)

    # Ensure that plot dir is created!
    PLOT_DIR.mkdir(exist_ok=True)

    now = datetime.utcnow()
    plotList = []
    logger.info(f"site: {site}")
    logger.info(f"span: {span}")

    spanString = span_dict[span]
    # load data for site
    siteData = dashboard.loadData(site, sites_dict)

    logger.info("Coercing `qartod_executed` to int for each test, then drop original variable.")
    siteData = coerce_qartod_executed_to_int(siteData)

    fileParams = sites_dict[site]['dataParameters'].strip('"').split(',')
    allVar = list(siteData.keys())
    # add qartod and qc flags to fileParams list
    qcStrings = ['_qartod_','_qc_']
    qcParams = [var for var in allVar if any(sub in var for sub in fileParams) if any(qc in var for qc in qcStrings)]
    fileParams = fileParams + qcParams
    # drop un-used variables from dataset
    dropList = [item for item in allVar if item not in fileParams]
    siteData = siteData.drop(dropList)

    logger.info(f"site date array: {siteData}")
    # extract parameters from multi-dimensional array
    if plotInstrument in multiParameter_dict.keys():
        siteData, fileParams = extractMulti(
            siteData, plotInstrument, multiParameter_dict, fileParams
        )

    if int(span) == 365:
        if len(siteData['time']) > decimationThreshold:
            # decimate data
            siteData_df = decimate.downsample(
                siteData, decimationThreshold, logger=logger
            )
            # turn dataframe into dataset
            del siteData
            gc.collect()
            siteData = xr.Dataset.from_dataframe(siteData_df, sparse=False)
            siteData = siteData.swap_dims({'index': 'time'})
            siteData = siteData.reset_coords()

    for param in paramList:
        logger.info(f"parameter: {param}")
        variableParams = variable_dict[param].strip('"').split(',')
        parameterList = [
            value for value in variableParams if value in fileParams
        ]
        if len(parameterList) != 1:
            logger.info("Error retriving parameter name...")
        else:
            Yparam = parameterList[0]
            # set up plotting parameters
            imageName_base = plotDir + site + '_' + param
            plotTitle = site + ' ' + param
            paramMin = float(variable_paramDict[param]['min'])
            paramMax = float(variable_paramDict[param]['max'])
            profile_paramMin = float(variable_paramDict[param]['profileMin'])
            profile_paramMax = float(variable_paramDict[param]['profileMax'])
            # default local range to standard range if not defined
            paramMin_local = paramMin
            paramMax_local = paramMax
            profile_paramMin_local = profile_paramMin
            profile_paramMax_local = profile_paramMax
            localRanges = str(localRange_dict[site][param])
            if not 'nan' in localRanges:
                localRange = literal_eval(localRanges)
                if 'local' in localRange:
                    paramMin_local = localRange['local'][0]
                    paramMax_local = localRange['local'][1]
                if 'local_profile' in localRange:
                    profile_paramMin_local = localRange['local_profile'][0]
                    profile_paramMax_local = localRange['local_profile'][1]

            yLabel = variable_paramDict[param]['label']

            # Load overlayData
            overlayData_clim = {}
            overlayData_grossRange = {}
            sensorType = site.split('-')[3][0:5].lower()
            (overlayData_grossRange, overlayData_clim) = dashboard.loadQARTOD(
                site, Yparam, sensorType, logger=logger
            )
            overlayData_near = {}
            # overlayData_near = loadNear(site)

            overlayData_anno = {}
            overlayData_anno = dashboard.loadAnnotations(site)

            if 'PROFILER' in plotInstrument:
                profileList = dashboard.loadProfiles(site)
                pressureParams = (
                    variable_dict['pressure'].strip('"').split(',')
                )
                pressureParamList = [
                    value for value in pressureParams if value in fileParams
                ]
                if len(pressureParamList) != 1:
                    logger.info("Error retriving pressure parameter!")
                else:
                    pressParam = pressureParamList[0]
                    paramData = siteData[[Yparam, pressParam]].chunk('auto')
                    flagParams = [item for item in qcParams if Yparam in item]
                    flagParams.extend((Yparam, pressParam))
                    overlayData_flag = siteData[flagParams].chunk('auto')
                    colorMap = 'cmo.' + variable_paramDict[param]['colorMap']
                    depthMinMax = (
                        sites_dict[site]['depthMinMax'].strip('"').split(',')
                    )
                    if 'None' not in depthMinMax:
                        yMin = int(depthMinMax[0])
                        yMax = int(depthMinMax[1])
                    plots = dashboard.plotProfilesGrid(
                        Yparam,
                        pressParam,
                        paramData,
                        plotTitle,
                        yLabel,
                        timeRef,
                        yMin,
                        yMax,
                        profile_paramMin,
                        profile_paramMax,
                        profile_paramMin_local,
                        profile_paramMax_local,
                        colorMap,
                        imageName_base,
                        overlayData_anno,
                        overlayData_clim,
                        overlayData_near,
                        span,
                        spanString,
                        profileList,
                        statusDict,
                        site,
                    )
                    plotList.append(plots)
                    plots = dashboard.plotProfilesScatter(
                        Yparam,
                        pressParam,
                        paramData,
                        plotTitle,
                        timeRef,
                        profile_paramMin,
                        profile_paramMax,
                        profile_paramMin_local,
                        profile_paramMax_local,
                        imageName_base,
                        overlayData_anno,
                        overlayData_clim,
                        overlayData_flag,
                        overlayData_near,
                        span,
                        spanString,
                        profileList,
                        statusDict,
                        site,
                    )
                    depths = sites_dict[site]['depths'].strip('"').split(',')
                    if 'Single' not in depths:
                        for profileDepth in depths:
                            paramData_depth = paramData[Yparam].where(
                                (int(profileDepth) < paramData[pressParam])
                                & (
                                    paramData[pressParam]
                                    < (int(profileDepth) + 0.5)
                                )
                            )
                            overlayData_flag_extract = overlayData_flag.where(
                                (int(profileDepth) < overlayData_flag[pressParam])
                                & (
                                    overlayData_flag[pressParam]
                                    < (int(profileDepth) + 0.5)
                                )
                            )
                            plotTitle_depth = (
                                plotTitle + ': ' + profileDepth + ' meters'
                            )
                            imageName_base_depth = (
                                imageName_base + '_' + profileDepth + 'meters'
                            )
                            if overlayData_clim:
                                overlayData_clim_extract = (
                                    dashboard.extractClim(
                                        timeRef, profileDepth, overlayData_clim
                                    )
                                )
                            else:
                                overlayData_clim_extract = pd.DataFrame()
                            plots = dashboard.plotScatter(
                                Yparam,
                                paramData_depth,
                                plotTitle_depth,
                                yLabel,
                                timeRef,
                                profile_paramMin,
                                profile_paramMax,
                                profile_paramMin_local,
                                profile_paramMax_local,
                                imageName_base_depth,
                                overlayData_anno,
                                overlayData_clim_extract,
                                overlayData_flag_extract,
                                overlayData_near,
                                'medium',
                                span,
                                spanString,
                                statusDict,
                                site,
                            )
                            plotList.append(plots)
            else:
                paramData = siteData[Yparam]
                flagParams = [item for item in qcParams if Yparam in item]
                flagParams.append(Yparam)
                overlayData_flag = siteData[flagParams].chunk('auto')

                if overlayData_clim:
                    overlayData_clim_extract = dashboard.extractClim(
                        timeRef, '0', overlayData_clim
                    )
                else:
                    overlayData_clim_extract = pd.DataFrame()
                # PLOT
                plots = dashboard.plotScatter(
                    Yparam,
                    paramData,
                    plotTitle,
                    yLabel,
                    timeRef,
                    paramMin,
                    paramMax,
                    paramMin_local,
                    paramMax_local,
                    imageName_base,
                    overlayData_anno,
                    overlayData_clim_extract,
                    overlayData_flag,
                    overlayData_near,
                    'small',
                    span,
                    spanString,
                    statusDict,
                    site,
                )
                plotList.append(plots)

            del paramData
            gc.collect()
    del siteData
    gc.collect()
    end = datetime.utcnow()
    elapsed = end - now
    logger.info(f"{site} finished plotting: Time elapsed ({elapsed})")
    return plotList


def organize_images(
    sync_to_s3=False, bucket_name=S3_BUCKET, fs_kwargs={}
):
    logger = select_logger()
    if sync_to_s3 is True:
        import fsspec
        S3FS = fsspec.filesystem('s3', **fs_kwargs)

        logger.info("collecting existing 'profile' files")
        existing_files = S3FS.ls(f's3://{bucket_name}/')
        files_to_delete = [file for file in existing_files if 'profile' in file]
        # The number of profiles changes so we want to delete old profile files.
        for f in files_to_delete:
            S3FS.rm(f)
        logger.info("'profile' files deleted")

    for i in PLOT_DIR.iterdir():
        if i.is_file():
            if i.suffix == '.png' or i.suffix == '.svg':
                fname = i.name
                subsite = fname.split('-')[0]

                subsite_dir = PLOT_DIR / subsite
                subsite_dir.mkdir(exist_ok=True)

                destination = subsite_dir / fname
                i.replace(destination)

                # Sync to s3
                if sync_to_s3 is True:
                    fs_path = '/'.join(
                        [bucket_name, PLOT_DIR.name, subsite_dir.name, fname]
                    )
                    if S3FS.exists(fs_path):
                        S3FS.rm(fs_path)
                    S3FS.put(str(destination.absolute()), fs_path)
            else:
                print(f"{i} is not a `png` or `svg` file ... skipping ...")
        else:
            print(f"{i} is not a file ... skipping ...")


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='QAQC Dashboard Plot Creator'
    )
    arg_parser.add_argument('--time', type=str, default='2016-01-01')
    arg_parser.add_argument(
        '--instrument',
        type=str,
        default='ctd-profiler',
        help=f"Choices {str(list(selection_mapping.keys()))}",
    )
    arg_parser.add_argument(
        '--span',
        type=str,
        default='7',
        help=f"Choices {str(list(span_dict.keys()))}",
    )
    arg_parser.add_argument('--threshold', type=int, default=1000000)

    return arg_parser.parse_args()


def main():
    from loguru import logger

    args = parse_args()

    # User options ...
    timeString = args.time
    timeRef = parser.parse(timeString)
    logger.add("logfile_create_dashboard_{time}.log")
    logger.info('Dashboard creation initiated')
    # Always make sure it gets created
    PLOT_DIR.mkdir(exist_ok=True)

    plotInstrument = selection_mapping[args.instrument]

    #  For each param in instrument, append to list of params in checkbox
    paramList = []
    for param in (
        instrument_dict[plotInstrument]['plotParameters']
        .replace('"', '')
        .split(',')
    ):
        paramList.append(param)

    dataList = []
    for key, values in sites_dict.items():
        if plotInstrument in sites_dict[key]['instrument']:
            dataList.append(key)

    now = datetime.utcnow()
    logger.info(f"======= Creation started at: {now.isoformat()} ======")
    for site in dataList:
        run_dashboard_creation(
            site,
            paramList,
            timeRef,
            plotInstrument,
            args.span,
            args.threshold,
            logger=logger,
        )
        # Organize pngs and svgs into folders
        organize_images()

    end = datetime.utcnow()
    logger.info(
        f"======= Creation finished at: {end.isoformat()}. Time elapsed ({(end - now)}) ======",
    )


if __name__ == "__main__":
    main()
