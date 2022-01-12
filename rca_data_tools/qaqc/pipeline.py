import datetime
import os
import warnings
from pathlib import Path
from prefect import task, Flow, Parameter
from prefect.storage import Docker
from prefect.run_configs import ECSRun
import prefect.engine.signals as prefect_signals

from rca_data_tools.qaqc.plots import (
    instrument_dict,
    organize_pngs,
    run_dashboard_creation,
)

HERE = Path(__file__).parent.absolute()


def register_flow(flow: Flow, project_name: str = 'rca-qaqc'):
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
def dashboard_creation_task(
    site, paramList, timeString, plotInstrument, span, threshold, logger
):
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
        prefect_signals.FAIL(message=f"PNG Creation Failed for {site}: {e}")


@task
def organize_pngs_task(
    plotList=[], fs_kwargs={}, sync_to_s3=False, s3_bucket='rca-qaqc'
):
    if len(plotList) > 0:
        organize_pngs(
            sync_to_s3=sync_to_s3, fs_kwargs=fs_kwargs, bucket_name=s3_bucket
        )
    else:
        prefect_signals.SKIP(message="No plots found to be organized.")


class QAQCPipeline:
    __dockerfile_path = HERE / "docker" / "Dockerfile"
    __prefect_directory = "/home/jovyan/prefect"

    def __init__(
        self,
        name,
        time='2020-06-30',
        threshold=1000000,
        cloud_run=False,
        prefect_project_name='rca-qaqc',
        s3_bucket='rca-qaqc',
        s3_sync=False,
        s3fs_kwargs={},
    ):
        self.name = name
        self.time = time
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
        self.site, self.plotInstrument, self.span = self.name.split('--')

    def __repr__(self):
        return self.name

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
    def image_info(self):
        tag = f"{self.name}.{self.created_dt:%Y%m%dT%H%M}"
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
        # TODO: Add schedule so it can cron away!
        with Flow(
            self.name, storage=self.storage, run_config=self.run_config
        ) as flow:
            # For dashboard png creation
            site_param = Parameter('site', default=self.site, required=False)
            timeString_param = Parameter(
                'timeString', default=self.time, required=False
            )
            paramList_param = Parameter(
                'paramList', default=self.parameters, required=False
            )
            plotInstrument_param = Parameter(
                'plotInstrument', default=self.plotInstrument, required=False
            )
            span_param = Parameter('span', default=self.span, required=False)
            threshold_param = Parameter(
                'threshold', default=self.threshold, required=False
            )
            logger_param = Parameter(
                'logger', default='prefect', required=False
            )

            # For organizing pngs
            fs_kwargs_param = Parameter(
                'fs_kwargs', default=self.s3fs_kwargs, required=False
            )
            sync_to_s3_param = Parameter(
                'sync_to_s3', default=self.s3_sync, required=False
            )
            s3_bucket_param = Parameter(
                's3_bucket', default=self.s3_bucket, required=False
            )

            plotList = dashboard_creation_task(
                site=site_param,
                paramList=paramList_param,
                timeString=timeString_param,
                plotInstrument=plotInstrument_param,
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

        self.flow = flow

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
            'labels': ['ecs-agent', 'ooi', 'prod'],
            'run_task_kwargs': {
                'cluster': 'prefectECSCluster',
                'launchType': 'FARGATE',
            },
            'env': {
                'GH_PAT': os.environ.get('GH_PAT', ''),
                'AWS_KEY': os.environ.get('AWS_KEY', ''),
                'AWS_SECRET': os.environ.get('AWS_SECRET', ''),
                'PREFECT__CLOUD__HEARTBEAT_MODE': 'thread',
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
