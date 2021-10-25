import argparse
import concurrent.futures
from datetime import datetime
from dateutil import parser
import gc
from loguru import logger
import pandas as pd
from pathlib import Path

from rca_data_tools.qaqc import dashboard as dashFunc

HERE = Path(__file__).parent.absolute()
PARAMS_DIR = HERE.joinpath('params')
PLOT_DIR = Path('QAQCplots')

# Always make sure it gets created
PLOT_DIR.mkdir(exist_ok=True)

selection_mapping = {'profiler': 'CTD-PROFILER', 'fixed': 'CTD-FIXED'}


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


def create_plot_for_depth(
    profileDepth,
    paramData,
    Yparam,
    pressParam,
    plotTitle,
    imageName_base,
    overlayData_clim,
    yLabel,
    profile_paramMin,
    profile_paramMax,
    overlayData_near,
    timeRef,
):
    print(f"Depth: {profileDepth}")
    paramData_depth = paramData[Yparam].where(
        (int(profileDepth) < paramData[pressParam])
        & (paramData[pressParam] < (int(profileDepth) + 0.5))
    )
    plotTitle_depth = plotTitle + ': ' + profileDepth + ' meters'
    imageName_base_depth = imageName_base + '_' + profileDepth + 'meters'
    if overlayData_clim:
        overlayData_clim_extract = dashFunc.extractClim(
            profileDepth, overlayData_clim, timeRef
        )
    else:
        overlayData_clim_extract = pd.DataFrame()
    dashFunc.plotScatter(
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
    )


def run_dashboard_creation(site, paramList, timeRef, plotInstrument):
    now = datetime.utcnow()
    plotList = []
    logger.info("site: {}", site)
    # load data for site
    siteData = dashFunc.loadData(site, sites_dict)
    fileParams = sites_dict[site]['dataParameters'].strip('"').split(',')
    for param in paramList:
        logger.info("paramter: {}", param)
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
            (overlayData_grossRange, overlayData_clim) = dashFunc.loadQARTOD(
                site, Yparam, sensorType
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
                    plots = dashFunc.plotProfilesGrid(
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
                                    dashFunc.extractClim(
                                        timeRef, profileDepth, overlayData_clim
                                    )
                                )
                            else:
                                overlayData_clim_extract = pd.DataFrame()
                            plots = dashFunc.plotScatter(
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
                            )
                            plotList.append(plots)
            else:
                paramData = siteData[Yparam]
                if overlayData_clim:
                    overlayData_clim_extract = dashFunc.extractClim(
                        timeRef, '0', overlayData_clim
                    )
                else:
                    overlayData_clim_extract = pd.DataFrame()
                # PLOT
                plots = dashFunc.plotScatter(
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
                )
                plotList.append(plots)

            del paramData
            gc.collect()
    del siteData
    gc.collect()
    end = datetime.utcnow()
    elapsed = end - now
    logger.info("{} finished plotting: Time elapsed ({})", site, str(elapsed))
    return plotList


def organize_pngs():
    for i in PLOT_DIR.iterdir():
        if i.is_file():
            fname = i.name
            subsite = fname.split('-')[0]

            subsite_dir = PLOT_DIR / subsite
            if not subsite_dir.exists():
                subsite_dir.mkdir()

            destination = subsite_dir / fname
            if not destination.exists():
                i.replace(destination)
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

    return arg_parser.parse_args()


def main():
    args = parse_args()

    # User options ...
    timeString = args.time
    timeRef = parser.parse(timeString)
    logger.add("logfile_create_dashboard_{time}.log")
    logger.info('Dashboard creation initiated')

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
    logger.info("======= Creation started at: {} ======", now.isoformat())

    logger.info(dataList)
    for site in dataList:
        sitePlotList = run_dashboard_creation(
            site, paramList, timeRef, plotInstrument
        )

    # Organize pngs into folders
    organize_pngs()
    end = datetime.utcnow()
    logger.info(
        "======= Creation finished at: {}. Time elapsed ({}) ======",
        end.isoformat(),
        (end - now),
    )


if __name__ == "__main__":
    main()
