# -*- coding: utf-8 -*-
"""pipeline.py

This module has been adapted to build prefect 2 flows. 

"""
import datetime
import os
from typing import Dict, Tuple, Optional, Any
import warnings
import argparse
import time
from pathlib import Path
from prefect import task, flow
#from prefect.storage import Docker
#from prefect.run_configs import ECSRun
#from prefect.tasks.prefect import create_flow_run
#import prefect.engine.signals as prefect_signals

# ** PREFECT 2 **
from prefect.states import Failed, Cancelled
from prefect.deployments import run_deployment

from rca_data_tools.qaqc.plots import (
    instrument_dict,
    sites_dict,
    span_dict,
)
from rca_data_tools.qaqc.flow import qaqc_pipeline_flow, S3_BUCKET

HERE = Path(__file__).parent.absolute()
#S3_BUCKET = 'ooi-rca-qaqc-prod'
#PROJECT_NAME = 'rca-qaqc'


# def register_flow(
#     flow: Flow, project_name: str = PROJECT_NAME
# ) -> Tuple[str, str]:
#     """
#     Register flow to prefect cloud.

#     Parameters
#     ----------
#     x : prefect.Flow
#         Prefect flow object to be registered.
#     y : str
#         Project name where the prefect flow should be registered to.
#         Defaults to `rca-qaqc`.

#     Returns
#     -------
#     tuple
#         (flow name, project name)

#     """
#     ready = False
#     while not ready:
#         # Keep trying to avoid docker registry interruptions
#         try:
#             flow.validate()
#             res = flow.register(project_name=project_name)
#             if isinstance(res, str):
#                 ready = True
#         except Exception as e:
#             warnings.warn(e)
#             ready = False
#     return flow.name, project_name


# @task
# def dashboard_creation_task(
#     site, 
#     timeString, 
#     span, 
#     threshold, 
#     #logger
#     ):
#     """
#     Prefect task for running dashboard creation
#     """
#     site_ds = sites_dict[site]
#     plotInstrument = site_ds['instrument']
#     paramList = (
#         instrument_dict[plotInstrument]['plotParameters']
#         .replace('"', '')
#         .split(',')
#     )
#     try:
#         plotList = run_dashboard_creation(
#             site,
#             paramList,
#             timeString,
#             plotInstrument,
#             span,
#             threshold,
#             #logger,
#         )
#         return plotList
#     except Exception as e:
#         # raise prefect_signals.FAIL(
#         #     message=f"PNG Creation Failed for {site}: {e}"
#         # )
#         return Failed(message=f"PNG Creation Failed for {site}: {e}")
        


# @task
# def organize_pngs_task(
#     plotList=[], fs_kwargs={}, sync_to_s3=False, s3_bucket=S3_BUCKET
# ):
#     """
#     Prefect task for organizing the plot pngs to their appropriate directories
#     """
#     if len(plotList) > 0:
#         organize_pngs(
#             sync_to_s3=sync_to_s3, fs_kwargs=fs_kwargs, bucket_name=s3_bucket
#         )
#     else:
#         #raise prefect_signals.SKIP(message="No plots found to be organized.")
#         return Cancelled(message="No plots found to be organized.")


# def create_flow(
#     name: str = "create_dashboard",
#     #storage: Optional[Docker] = None,
#     #run_config: Optional[ECSRun] = None,
#     schedule: Optional[Any] = None,
# ):
#     """
#     Create prefect flow for plot creation tasks.

#     Parameters
#     ----------
#     name : str
#         The name of the prefect flow.
#     storage : prefect.storage.Docker, optional
#         Prefect storage where the flow and tasks code should be stored.
#     run_config: prefect.storage.ECSRun, optional
#         Prefect run configurations.
#     schedule: optional
#         Prefect schedule.
#         **Note: Currently not used**

#     Returns
#     -------
#     prefect.Flow
#         Prefect flow objects with specified name,
#         storage, run configs, and schedule.

#     """
    # now = datetime.datetime.utcnow()
    # TODO: Add schedule so it can cron away!
    # with Flow(
    #     name, storage=storage, run_config=run_config, schedule=schedule
    # ) as flow:
    #     # For dashboard png creation
    #     site_param = Parameter(
    #         'site', default='CE02SHBP-LJ01D-06-CTDBPN106', required=True
    #     )
    #     timeString_param = Parameter(
    #         'timeString', default=now.strftime('%Y-%m-%d'), required=False
    #     )
    #     span_param = Parameter('span', default='1', required=False)
    #     threshold_param = Parameter(
    #         'threshold', default=1000000, required=False
    #     )
    #     logger_param = Parameter('logger', default='prefect', required=False)

    #     # For organizing pngs
    #     fs_kwargs_param = Parameter('fs_kwargs', default={}, required=False)
    #     sync_to_s3_param = Parameter(
    #         'sync_to_s3', default=False, required=False
    #     )
    #     s3_bucket_param = Parameter(
    #         's3_bucket', default=S3_BUCKET, required=False
    #     )

# now = datetime.datetime.utcnow()
# default_run_config = {} #TODO something like this for run configs 

# @flow
# def qaqc_pipeline_flow(
#     name: str='create dashboard',
#     #run_config: dict=default_run_config, #TODO something like this for run configs?
#     site_param: str='CE02SHBP-LJ01D-06-CTDBPN106',
#     timeString_param: str=now.strftime('%Y-%m-%d'),
#     span_param: str='1',
#     threshold_param: int=1000000,
#     #logger_param
#     # For organizing pngs
#     fs_kwargs_param: dict={},
#     sync_to_s3_param: bool=False,
#     s3_bucket_param: str=S3_BUCKET,
#     ):

#     # Run dashboard creation task
#     plotList = dashboard_creation_task(
#         site=site_param,
#         timeString=timeString_param,
#         span=span_param,
#         threshold=threshold_param,
#         #logger=logger_param,
#     )

#     # Run organize pngs task
#     organize_pngs_task(
#         plotList=plotList,
#         sync_to_s3=sync_to_s3_param,
#         fs_kwargs=fs_kwargs_param,
#         s3_bucket=s3_bucket_param,
#     )
#return flow


class QAQCPipeline:
    """
    QAQC Pipeline Class to create Pipeline for specified site, time, and span.


    """
    __dockerfile_path = HERE / "docker" / "Dockerfile" #TODO is this necessary?
    __prefect_directory = "/home/jovyan/prefect"

    def __init__(
        self,
        site=None,
        time='2020-06-30',
        span='1',
        threshold=1000000,
        cloud_run=False,
        prefect_project_name='rca-qaqc',
        s3_bucket=S3_BUCKET,
        s3_sync=False,
        s3fs_kwargs={},
    ):
        self.site = site
        self.time = time
        self.span = span
        self.threshold = threshold
        self._cloud_run = cloud_run
        self.prefect_project_name = prefect_project_name
        self.s3_bucket = s3_bucket
        self.s3_sync = s3_sync
        self.s3fs_kwargs = s3fs_kwargs
        self._site_ds = {}

        self.__setup() #TODO
        #self.__setup_flow() #TODO

    def __setup(self):
        self.created_dt = datetime.datetime.utcnow()
        if self.site is not None:
            if self.site not in sites_dict:
                raise ValueError(
                    f"{self.site} is not available. Available sites {','.join(list(sites_dict.keys()))}"  # noqa
                )
            self._site_ds = sites_dict[self.site]
            self.plotInstrument = self._site_ds.get('instrument', None)
            if self.span not in span_dict:
                raise ValueError(
                    f"{self.span} not valid. Must be {','.join(list(span_dict.keys()))}"  # noqa
                )
            self.name = f"{self.site}--{self.plotInstrument}--{self.span}"
        else:
            self.name = "No site"

    def __repr__(self):
        return f"<{self.name}>"

    @property
    def cloud_run(self):
        return self._cloud_run

    @cloud_run.setter
    def cloud_run(self, cr):
        self._cloud_run = cr
        #self.__setup_flow()

    @property
    def parameters(self):
        """
        OOI plot parameters
        """
        return (
            instrument_dict[self.plotInstrument]['plotParameters']
            .replace('"', '')
            .split(',')
        )

    @property
    def flow_parameters(self):
        """
        Prefect flow parameters
        """
        return {
            'site': self.site,
            'timeString': self.time,
            'span': self.span,
            'threshold': self.threshold,
            'fs_kwargs': self.s3fs_kwargs,
            'sync_to_s3': self.s3_sync,
            's3_bucket': self.s3_bucket,
        }

    @property
    def image_info(self):
        """
        Docker Image Info
        """
        tag = f"{self.created_dt:%Y%m%dT%H%M}"
        registry = "cormorack"
        repo = "qaqc-dashboard"
        return {
            'tag': tag,
            'registry': registry,
            'repo': repo,
            'url': f"{registry}/{repo}:{tag}",
        }

    @property
    def dockerfile(self):
        """
        Dockerfile content
        """
        return self.__dockerfile_path.read_text(encoding='utf-8')

    # @property
    # def storage(self):
    #     """
    #     Docker Storage Option
    #     """
    #     if self.cloud_run is True:
    #         storage_options = self.docker_storage_options()
    #         return Docker(**storage_options)
    #     return

    # @property
    # def run_config(self):
    #     """
    #     ECS Run Configuration
    #     """
    #     if self.cloud_run is True:
    #         # NOTE: As of 4/28/2022 instance resources is not used at this time
    #         resources = self._parse_resources()
    #         run_config = self.ecs_run_options(
    #             cpu=resources.get('cpu', None),
    #             memory=resources.get('memory', None),
    #         )
    #         return ECSRun(**run_config)
    #     return

    @staticmethod
    def _get_resource_values(resource: str) -> Dict:
        span_configs = [sp.split('::') for sp in resource.split(',')]
        return dict(span_configs)

    def _parse_resources(self) -> Dict:
        cpu = self._site_ds.get('cpu', None)
        memory = self._site_ds.get('memory', None)
        instance = self._site_ds.get('instance', None)
        cpu_spans = {}
        memory_spans = {}
        instance_spans = {}

        if isinstance(cpu, str):
            cpu_spans = self._get_resource_values(cpu)

        if isinstance(memory, str):
            memory_spans = self._get_resource_values(memory)

        if isinstance(instance, str):
            instance_spans = self._get_resource_values(instance)

        return {
            'cpu': cpu_spans.get(self.span, None),
            'memory': memory_spans.get(self.span, None),
            'instance': instance_spans.get(self.span, None),
        }

    # def __setup_flow(self):
    #     self.flow = create_flow()
    #     self.flow.storage = self.storage
    #     self.flow.run_config = self.run_config

    def run(self, parameters=None):
        """
        Runs the flow either in the cloud or locally.
        """
        from loguru import logger

        if self.site is None:
            raise ValueError("No site found. Please provide site.")
        if parameters is None:
            parameters = self.flow_parameters
    
        logger.info(f"parameters set to: {parameters}!")
        if self.cloud_run is True:
            # create_flow_run.run(
            #     flow_name=self.flow.name,
            #     project_name=PROJECT_NAME,
            #     parameters=parameters,
            #     run_config=self.run_config,
            #     run_name=self.name,
            # )
            run_name = "-".join([str(self.site), str(self.time), str(self.threshold), str(self.span), "flow_run"])
            run_deployment(
                name="qaqc-pipeline-flow/4vcpu_16gb",
                parameters=parameters,
                flow_run_name=run_name,
                timeout=10 #TODO timeout might need to be increase if we have race condition errors
            )
        else:
            qaqc_pipeline_flow()

    # def docker_storage_options( #TODO can this method be deprecated?
    #     self,
    #     registry_url=None,
    #     image_name=None,
    #     image_tag=None,
    #     dockerfile=None,
    #     prefect_directory=None,
    #     python_dependencies=None,
    #     **kwargs,
    # ) -> Dict[str, Any]:
    #     """
    #     Create default docker storage options dictionary
    #     """
    #     default_dependencies = [
    #         'git+https://github.com/OOI-CabledArray/rca-data-tools.git@main'
    #     ]
    #     storage_options = {
    #         'registry_url': self.image_info['registry']
    #         if registry_url is None
    #         else registry_url,
    #         'image_name': self.image_info['repo']
    #         if image_name is None
    #         else image_name,
    #         'image_tag': self.image_info['tag']
    #         if image_tag is None
    #         else image_tag,
    #         'dockerfile': self.__dockerfile_path
    #         if dockerfile is None
    #         else dockerfile,
    #         'prefect_directory': self.__prefect_directory
    #         if prefect_directory is None
    #         else prefect_directory,
    #         'python_dependencies': default_dependencies
    #         if python_dependencies is None
    #         else python_dependencies,
    #     }
    #     return dict(**storage_options, **kwargs)

    # def ecs_run_options( #TODO can this method also be deprecated?
            
    #     self,
    #     cpu=None,
    #     memory=None,
    #     labels=None,
    #     task_role_arn=None,
    #     run_task_kwargs=None,
    #     execution_role_arn=None,
    #     env=None,
    #     **kwargs,
    # ) -> Dict[str, Any]:
    #     """
    #     Create default ecs run configuration options dictionary
    #     """
    #     defaults = {
    #         'cpu': '4 vcpu',
    #         'memory': '30 GB',
    #         'labels': ['rca', 'prod'],
    #         'run_task_kwargs': {
    #             'cluster': 'prefectECSCluster',
    #             'launchType': 'FARGATE',
    #             'tags': [
    #                 {'key': 'Owner', 'value': 'RCA Data Team'},
    #                 {'key': 'Name', 'value': 'QAQC Dashboard Pipeline'},
    #                 {'key': 'Project', 'value': 'Regional Cabled Array'},
    #                 {'key': 'Environment', 'value': 'prod'},
    #             ],
    #         },
    #         'env': {
    #             'GH_PAT': os.environ.get('GH_PAT', ''),
    #             'AWS_KEY': os.environ.get('AWS_KEY', ''),
    #             'AWS_SECRET': os.environ.get('AWS_SECRET', ''),
    #             'PREFECT__CLOUD__HEARTBEAT_MODE': 'thread',
    #             'AWS_RETRY_MODE': os.environ.get('AWS_RETRY_MODE', 'adaptive'),
    #             'AWS_MAX_ATTEMPTS': os.environ.get('AWS_MAX_ATTEMPTS', '100'),
    #         },
    #     }
    #     run_options = {
    #         'env': defaults['env'] if env is None else env,
    #         'cpu': defaults['cpu'] if cpu is None else cpu,
    #         'memory': defaults['memory'] if memory is None else memory,
    #         'labels': defaults['labels'] if labels is None else labels,
    #         'task_role_arn': os.environ.get(
    #             'TASK_ROLE_ARN',
    #             '',
    #         )
    #         if task_role_arn is None
    #         else task_role_arn,
    #         'execution_role_arn': os.environ.get(
    #             'EXECUTION_ROLE_ARN',
    #             '',
    #         )
    #         if execution_role_arn is None
    #         else execution_role_arn,
    #         'run_task_kwargs': defaults['run_task_kwargs']
    #         if run_task_kwargs is None
    #         else run_task_kwargs,
    #     }
    #     return dict(**run_options, **kwargs)


def parse_args():
    arg_parser = argparse.ArgumentParser(description='QAQC Pipeline Register')

    arg_parser.add_argument('--register', action="store_true")
    arg_parser.add_argument('--all', action="store_true")
    arg_parser.add_argument('--run', action="store_true")
    arg_parser.add_argument('--cloud', action="store_true")
    arg_parser.add_argument('--s3-sync', action="store_true")
    arg_parser.add_argument('--site', type=str, default=None)
    arg_parser.add_argument('--time', type=str, default='2020-06-30')
    arg_parser.add_argument(
        '--s3-bucket',
        type=str,
        default=S3_BUCKET,
        help="S3 Bucket to store the plots.",
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
    if args.all is True:
        # Creates pipeline objects for all the sites
        # if run is specified, will actually run the pipeline
        # in prefect cloud.
        now = datetime.datetime.utcnow()
        for key in sites_dict.keys():
            logger.info(f"creating pipeline instance for site: {key}")
            pipeline = QAQCPipeline(
                site=key,
                cloud_run=args.cloud,
                s3_sync=args.s3_sync,
                s3_bucket=args.s3_bucket,
                time=now.strftime("%Y-%m-%d"),
                span=args.span,
                threshold=args.threshold,
            )
            logger.info(f"{pipeline.name} created.")
            if args.run is True:
                pipeline.run()
            # Add 10s delay for each run
            time.sleep(10)
    # else:
    #     # Creates only one pipeline
    #     # This is used for registration or testing only
    #     pipeline = QAQCPipeline(
    #         site=args.site,
    #         cloud_run=args.cloud,
    #         s3_sync=args.s3_sync,
    #         s3_bucket=args.s3_bucket,
    #         time=args.time,
    #         span=args.span,
    #         threshold=args.threshold,
    #     )

    #     if args.register is True:
    #         # logger.info(f"Registering pipeline {pipeline.flow.name}.")
    #         # register_flow(pipeline.flow)
    #         logger.warning("Joe thinks this argument is no longer necessary")

    #     if args.run is True:
    #         pipeline.run()
