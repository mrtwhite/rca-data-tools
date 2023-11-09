# -*- coding: utf-8 -*-
"""pipeline.py

This module contains the qaqc_pipeline entry point: main() and the QAQCPipeline
class. This class interfaces with the cli entry point to orchestrate a pipeline 
with prefect 2 which uses the zarr files in the ooi-data-prod s3 bucket to generate 
plots as pngs. These plots are viewable throug the frontend web app in QAQC_dashboard.

Prefect 2 @flow and @task decorated functions are found in flow.py

"""
import datetime
from typing import Dict
import argparse
import time
from pathlib import Path

from prefect.deployments import run_deployment

from rca_data_tools.qaqc.plots import (
    instrument_dict,
    sites_dict,
    span_dict,
)
from rca_data_tools.qaqc.compute_constants import COMPUTE_EXCEPTIONS
from rca_data_tools.qaqc.flow import qaqc_pipeline_flow, S3_BUCKET

HERE = Path(__file__).parent.absolute()


class QAQCPipeline:
    """
    QAQC Pipeline Class to create Pipeline for specified site, time, and span.


    """
    __dockerfile_path = HERE / "docker" / "Dockerfile" #TODO is this necessary?

    def __init__(
        self,
        site=None,
        time='2020-06-30',
        span='1',
        threshold=1_000_000,
        cloud_run=True,
        s3_bucket=S3_BUCKET,
        s3_sync=True,
        s3fs_kwargs={},
    ):
        self.site = site
        self.time = time
        self.span = span
        self.threshold = threshold
        self._cloud_run = cloud_run
        self.s3_bucket = s3_bucket
        self.s3_sync = s3_sync
        self.s3fs_kwargs = s3fs_kwargs
        self._site_ds = {}

        self.__setup()
        #self.__setup_flow()

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
            run_name = "-".join([str(self.site), str(self.time), str(self.threshold), str(self.span), "flow_run"])
            # IMPORTANT run_deployment determines the infrastructure and resources for each flow_run
            if self.site in COMPUTE_EXCEPTIONS and self.span in COMPUTE_EXCEPTIONS[self.site]:

                deployment_name = f"qaqc-pipeline-flow/{COMPUTE_EXCEPTIONS[self.site][self.span]}"
                logger.warning(f"{self.site} with span {self.span} requires additional compute resources, creating flow_run from {deployment_name} instead of default")
                run_deployment(
                    name=deployment_name,
                    parameters=parameters,
                    flow_run_name=run_name,
                    timeout=10
                )
            # otherwise run the default deployment with default compute resources        
            else:
                run_deployment(
                    name="qaqc-pipeline-flow/4vcpu_30gb",
                    parameters=parameters,
                    flow_run_name=run_name,
                    timeout=10 #TODO timeout might need to be increase if we have race condition errors
                )
        else:
            qaqc_pipeline_flow()


def parse_args():
    arg_parser = argparse.ArgumentParser(description='QAQC Pipeline Register')

    arg_parser.add_argument('--register', action="store_true")
    arg_parser.add_argument('--all', action="store_true")
    arg_parser.add_argument('--run', action="store_true")
    arg_parser.add_argument('--cloud', action="store_true")
    arg_parser.add_argument('--s3-sync', action="store_true")
    arg_parser.add_argument('--site', type=str, default=None)
    arg_parser.add_argument('--time', type=str, default=None)
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
    arg_parser.add_argument('--threshold', type=int, default=5000000)

    return arg_parser.parse_args()


def main():
    from loguru import logger

    args = parse_args()
    now = datetime.datetime.utcnow()
    if args.all is True:
        # Creates pipeline objects for all the sites
        # if run is specified, will actually run the pipeline
        # in prefect cloud.
        for key in sites_dict.keys():
            logger.info(f"creating pipeline instance for site: {key}")
            pipeline = QAQCPipeline(
                site=key,
                time=now.strftime("%Y-%m-%d"),
                span=args.span,
                threshold=args.threshold,
                cloud_run=args.cloud,
                s3_bucket=args.s3_bucket,
                s3_sync=args.s3_sync,
            )
            logger.info(f"{pipeline.name} created.")
            if args.run is True:
                pipeline.run()
            # Add 20s delay for each run
            time.sleep(20)
    else:
        # Creates only one pipeline
        # This may be useful for testing
        pipeline = QAQCPipeline(
            site=args.site,
            time=now.strftime("%Y-%m-%d"),
            span=args.span,
            threshold=args.threshold,
            cloud_run=args.cloud,
            s3_bucket=args.s3_bucket,
            s3_sync=args.s3_sync,
        )
        logger.info(f"{pipeline.name} created")

        if args.run is True:
            pipeline.run()
