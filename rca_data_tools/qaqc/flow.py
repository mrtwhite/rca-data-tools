import datetime
from prefect import task, flow
from prefect.states import Failed, Cancelled

from rca_data_tools.qaqc.plots import (
    instrument_dict,
    organize_pngs,
    run_dashboard_creation,
    sites_dict,
)

S3_BUCKET = 'ooi-rca-qaqc-prod'

@task
def dashboard_creation_task(
    site, 
    timeString, 
    span, 
    threshold, 
    #logger
    ):
    """
    Prefect task for running dashboard creation
    """
    site_ds = sites_dict[site]
    plotInstrument = site_ds['instrument']
    paramList = (
        instrument_dict[plotInstrument]['plotParameters']
        .replace('"', '')
        .split(',')
    )

    plotList = run_dashboard_creation(
        site,
        paramList,
        timeString,
        plotInstrument,
        span,
        threshold,
        #logger,
    )
    return plotList
    # except Exception as e:
        # raise prefect_signals.FAIL(
        #     message=f"PNG Creation Failed for {site}: {e}"
        # )
        # return Failed(message=f"PNG Creation Failed for {site}: {e}")
        

@task
def organize_pngs_task(
    plotList=[], fs_kwargs={}, sync_to_s3=False, s3_bucket=S3_BUCKET
):
    """
    Prefect task for organizing the plot pngs to their appropriate directories
    """
    if len(plotList) > 0:
        organize_pngs(
            sync_to_s3=sync_to_s3, fs_kwargs=fs_kwargs, bucket_name=s3_bucket
        )
    else:
        #raise prefect_signals.SKIP(message="No plots found to be organized.")
        return Cancelled(message="No plots found to be organized.")
    

now = datetime.datetime.utcnow()

@flow
def qaqc_pipeline_flow(
    name: str='create dashboard',
    #run_config: dict=default_run_config, #TODO something like this for run configs?
    site_param: str='CE02SHBP-LJ01D-06-CTDBPN106',
    timeString_param: str=now.strftime('%Y-%m-%d'),
    span_param: str='1',
    threshold_param: int=1000000,
    #logger_param
    # For organizing pngs
    fs_kwargs_param: dict={},
    sync_to_s3_param: bool=False,
    s3_bucket_param: str=S3_BUCKET,
    ):

    # Run dashboard creation task
    plotList = dashboard_creation_task(
        site=site_param,
        timeString=timeString_param,
        span=span_param,
        threshold=threshold_param,
        #logger=logger_param,
    )

    # Run organize pngs task
    organize_pngs_task(
        plotList=plotList,
        sync_to_s3=sync_to_s3_param,
        fs_kwargs=fs_kwargs_param,
        s3_bucket=s3_bucket_param,
    )
