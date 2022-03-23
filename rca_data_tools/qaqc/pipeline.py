import datetime
import os
import warnings
import argparse
import time
from pathlib import Path
from prefect import task, Flow, Parameter
from prefect.storage import Docker
from prefect.run_configs import ECSRun
from prefect.tasks.prefect import create_flow_run
import prefect.engine.signals as prefect_signals


from rca_data_tools.qaqc.plots import (
    instrument_dict,
    organize_pngs,
    run_dashboard_creation,
    sites_dict,
    span_dict,
)

HERE = Path(__file__).parent.absolute()
S3_BUCKET = 'qaqc.ooica.net'
PROJECT_NAME = 'rca-qaqc'


def register_flow(flow: Flow, project_name: str = PROJECT_NAME):
    ready = False
    while not ready:
        # Keep trying to avoid docker registry interruptions
        try:
            flow.validate()
            res = flow.register(project_name=project_name)
            if isinstance(res, str):
                ready = True
        except Exception as e:
            warnings.warn(e)
            ready = False
    return flow.name, project_name


@task
def dashboard_creation_task(site, timeString, span, threshold, logger):
    site_ds = sites_dict[site]
    plotInstrument = site_ds['instrument']
    paramList = (
        instrument_dict[plotInstrument]['plotParameters']
        .replace('"', '')
        .split(',')
    )
    try:
        plotList = run_dashboard_creation(
            site,
            paramList,
            timeString,
            plotInstrument,
            span,
            threshold,
            logger,
        )
        return plotList
    except Exception as e:
        raise prefect_signals.FAIL(
            message=f"PNG Creation Failed for {site}: {e}"
        )


@task
def organize_pngs_task(
    plotList=[], fs_kwargs={}, sync_to_s3=False, s3_bucket=S3_BUCKET
):
    if len(plotList) > 0:
        organize_pngs(
            sync_to_s3=sync_to_s3, fs_kwargs=fs_kwargs, bucket_name=s3_bucket
        )
    else:
        raise prefect_signals.SKIP(message="No plots found to be organized.")


def create_flow(
    name="create_dashboard", storage=None, run_config=None, schedule=None
):
    now = datetime.datetime.utcnow()
    # TODO: Add schedule so it can cron away!
    with Flow(
        name, storage=storage, run_config=run_config, schedule=schedule
    ) as flow:
        # For dashboard png creation
        site_param = Parameter(
            'site', default='CE02SHBP-LJ01D-06-CTDBPN106', required=True
        )
        timeString_param = Parameter(
            'timeString', default=now.strftime('%Y-%m-%d'), required=False
        )
        span_param = Parameter('span', default='1', required=False)
        threshold_param = Parameter(
            'threshold', default=1000000, required=False
        )
        logger_param = Parameter('logger', default='prefect', required=False)

        # For organizing pngs
        fs_kwargs_param = Parameter('fs_kwargs', default={}, required=False)
        sync_to_s3_param = Parameter(
            'sync_to_s3', default=False, required=False
        )
        s3_bucket_param = Parameter(
            's3_bucket', default=S3_BUCKET, required=False
        )

        plotList = dashboard_creation_task(
            site=site_param,
            timeString=timeString_param,
            span=span_param,
            threshold=threshold_param,
            logger=logger_param,
        )
        organize_pngs_task(
            plotList=plotList,
            sync_to_s3=sync_to_s3_param,
            fs_kwargs=fs_kwargs_param,
            s3_bucket=s3_bucket_param,
        )
    return flow


class QAQCPipeline:
    __dockerfile_path = HERE / "docker" / "Dockerfile"
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

        self.__setup()
        self.__setup_flow()

    def __setup(self):
        self.created_dt = datetime.datetime.utcnow()
        if self.site is not None:
            if self.site not in sites_dict:
                raise ValueError(
                    f"{self.site} is not available. Available sites {','.join(list(sites_dict.keys()))}"  # noqa
                )
            site_ds = sites_dict[self.site]
            self.plotInstrument = site_ds['instrument']
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
        self.__setup_flow()

    @property
    def parameters(self):
        return (
            instrument_dict[self.plotInstrument]['plotParameters']
            .replace('"', '')
            .split(',')
        )

    @property
    def flow_parameters(self):
        return {
            'site': self.site,
            'timeString': self.time,
            'threshold': self.threshold,
            'span': self.span,
            'fs_kwargs': self.s3fs_kwargs,
            'sync_to_s3': self.s3_sync,
            's3_bucket': self.s3_bucket,
        }

    @property
    def image_info(self):
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
        return self.__dockerfile_path.read_text(encoding='utf-8')

    @property
    def storage(self):
        if self.cloud_run is True:
            storage_options = self.docker_storage_options()
            return Docker(**storage_options)
        return

    @property
    def run_config(self):
        if self.cloud_run is True:
            run_config = self.ecs_run_options()
            return ECSRun(**run_config)
        return

    def __setup_flow(self):
        self.flow = create_flow()
        self.flow.storage = self.storage
        self.flow.run_config = self.run_config

    def run(self, parameters=None):
        if self.site is None:
            raise ValueError("No site found. Please provide site.")
        if parameters is None:
            parameters = self.flow_parameters

        if self.cloud_run is True:
            create_flow_run.run(
                flow_name=self.flow.name,
                project_name=PROJECT_NAME,
                parameters=parameters,
                run_config=self.run_config,
                run_name=self.name,
            )
        else:
            self.flow.run(parameters=parameters)

    def docker_storage_options(
        self,
        registry_url=None,
        image_name=None,
        image_tag=None,
        dockerfile=None,
        prefect_directory=None,
        python_dependencies=None,
        **kwargs,
    ):
        default_dependencies = [
            'git+https://github.com/OOI-CabledArray/rca-data-tools.git@main'
        ]
        storage_options = {
            'registry_url': self.image_info['registry']
            if registry_url is None
            else registry_url,
            'image_name': self.image_info['repo']
            if image_name is None
            else image_name,
            'image_tag': self.image_info['tag']
            if image_tag is None
            else image_tag,
            'dockerfile': self.__dockerfile_path
            if dockerfile is None
            else dockerfile,
            'prefect_directory': self.__prefect_directory
            if prefect_directory is None
            else prefect_directory,
            'python_dependencies': default_dependencies
            if python_dependencies is None
            else python_dependencies,
        }
        return dict(**storage_options, **kwargs)

    def ecs_run_options(
        self,
        cpu=None,
        memory=None,
        labels=None,
        task_role_arn=None,
        run_task_kwargs=None,
        execution_role_arn=None,
        env=None,
        **kwargs,
    ):
        defaults = {
            'cpu': '4 vcpu',
            'memory': '30 GB',
            'labels': ['rca', 'prod'],
            'run_task_kwargs': {
                'cluster': 'prefectECSCluster',
                'launchType': 'FARGATE',
            },
            'env': {
                'GH_PAT': os.environ.get('GH_PAT', ''),
                'AWS_KEY': os.environ.get('AWS_KEY', ''),
                'AWS_SECRET': os.environ.get('AWS_SECRET', ''),
                'PREFECT__CLOUD__HEARTBEAT_MODE': 'thread',
                'AWS_RETRY_MODE': os.environ.get('AWS_RETRY_MODE', 'adaptive'),
                'AWS_MAX_ATTEMPTS': os.environ.get('AWS_MAX_ATTEMPTS', '100'),
            },
        }
        run_options = {
            'env': defaults['env'] if env is None else env,
            'cpu': defaults['cpu'] if cpu is None else cpu,
            'memory': defaults['memory'] if memory is None else memory,
            'labels': defaults['labels'] if labels is None else labels,
            'task_role_arn': os.environ.get(
                'TASK_ROLE_ARN',
                '',
            )
            if task_role_arn is None
            else task_role_arn,
            'execution_role_arn': os.environ.get(
                'EXECUTION_ROLE_ARN',
                '',
            )
            if execution_role_arn is None
            else execution_role_arn,
            'run_task_kwargs': defaults['run_task_kwargs']
            if run_task_kwargs is None
            else run_task_kwargs,
        }
        return dict(**run_options, **kwargs)


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
        now = datetime.datetime.utcnow()
        for key in sites_dict.keys():
            pipeline = QAQCPipeline(
                site=key,
                cloud_run=args.cloud,
                s3_sync=args.s3_sync,
                time=now.strftime("%Y-%m-%d"),
                span=args.span,
                threshold=args.threshold,
            )
            logger.info(f"{pipeline.name} created.")
            if args.run is True:
                pipeline.run()
            # Add 10s delay for each run
            time.sleep(10)
    else:
        pipeline = QAQCPipeline(
            site=args.site,
            cloud_run=args.cloud,
            s3_sync=args.s3_sync,
            time=args.time,
            span=args.span,
            threshold=args.threshold,
        )

        if args.register is True:
            logger.info(f"Registering pipeline {pipeline.flow.name}.")
            register_flow(pipeline.flow)

        if args.run is True:
            pipeline.run()
