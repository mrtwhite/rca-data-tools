import argparse
import concurrent.futures
from datetime import datetime
from dateutil import parser
import gc
import pandas as pd
from pathlib import Path
import xarray as xr

from rca_data_tools.qaqc import dashboard
from rca_data_tools.qaqc import decimate
from rca_data_tools.qaqc import index

HERE = Path(__file__).parent.absolute()
PARAMS_DIR = HERE.joinpath('params')
PLOT_DIR = Path('QAQCplots')

selection_mapping = {'profiler': 'CTD-PROFILER', 'fixed': 'CTD-FIXED'}
span_dict = {'1': 'day', '7': 'week', '30': 'month', '365': 'year'}


# create dictionary of sites key for filePrefix, nearestNeighbors
sites_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('sitesDictionaryPanel.csv'))
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

plotDir = str(PLOT_DIR) + '/'


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
    logger=None,
):
    if logger == 'prefect':
        import prefect

        logger = prefect.context.get("logger")
    else:
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
    fileParams = sites_dict[site]['dataParameters'].strip('"').split(',')
    # drop un-used variables from dataset
    allVar = list(siteData.keys())
    dropList = [item for item in allVar if item not in fileParams]
    siteData = siteData.drop(dropList)
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
            paramMin = variable_paramDict[param]['min']
            paramMax = variable_paramDict[param]['max']
            profile_paramMin = variable_paramDict[param]['profileMin']
            profile_paramMax = variable_paramDict[param]['profileMax']
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

            if 'PROFILER' in plotInstrument:
                # TODO extract profiles???
                profile_paramMin = variable_paramDict[param]['profileMin']
                profile_paramMax = variable_paramDict[param]['profileMax']
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
                    paramData = siteData[[Yparam, pressParam]]
                    colorMap = 'cmo.' + variable_paramDict[param]['colorMap']
                    depthMinMax = (
                        sites_dict[site]['depthMinMax'].strip('"').split(',')
                    )
                    if 'None' not in depthMinMax:
                        yMin = int(depthMinMax[0])
                        yMax = int(depthMinMax[1])
                    plots = dashboard.plotProfilesGrid(
                        Yparam,
                        paramData,
                        plotTitle,
                        yLabel,
                        timeRef,
                        yMin,
                        yMax,
                        profile_paramMin,
                        profile_paramMax,
                        colorMap,
                        imageName_base,
                        overlayData_clim,
                        overlayData_near,
                        span,
                        spanString,
                    )
                    plotList.append(plots)
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
                                imageName_base_depth,
                                overlayData_clim_extract,
                                overlayData_near,
                                'medium',
                                span,
                                spanString,
                            )
                            plotList.append(plots)
            else:
                paramData = siteData[Yparam]
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
                    imageName_base,
                    overlayData_clim_extract,
                    overlayData_near,
                    'small',
                    span,
                    spanString,
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


def organize_pngs(sync_to_s3=False, bucket_name='rca-qaqc', fs_kwargs={}):
    if sync_to_s3 is True:
        import fsspec

        S3FS = fsspec.filesystem('s3', **fs_kwargs)

    for i in PLOT_DIR.iterdir():
        if i.is_file():
            if '.png' in str(i):
                fname = i.name
                subsite = fname.split('-')[0]

                subsite_dir = PLOT_DIR / subsite
                subsite_dir.mkdir(exist_ok=True)

                destination = subsite_dir / fname
                i.replace(destination)

                # Sync to s3
                if sync_to_s3 is True:
                    fs_path = '/'.join([bucket_name, subsite_dir.name, fname])
                    if S3FS.exists(fs_path):
                        S3FS.rm(fs_path)
                    S3FS.put(str(destination.absolute()), fs_path)
            else:
                print(f"{i} is not an image file ... skipping ...")
        else:
            print(f"{i} is not a file ... skipping ...")


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='QAQC Dashboard Plot Creator'
    )
    arg_parser.add_argument('--time', type=str, default='2020-06-30')
    arg_parser.add_argument(
        '--instrument',
        type=str,
        default='profiler',
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
        # Organize pngs into folders
        organize_pngs()

    index.create_local_index()

    end = datetime.utcnow()
    logger.info(
        f"======= Creation finished at: {end.isoformat()}. Time elapsed ({(end - now)}) ======",
    )


if __name__ == "__main__":
    main()
