import argparse
import json
import fsspec
from rca_data_tools.qaqc.plots import PLOT_DIR as PLOTSDIR
from rca_data_tools.qaqc.notes import HITL_NOTES_DIR as HITLDIR
from rca_data_tools.qaqc.pipeline import S3_BUCKET

INDEX_FILE = 'index.json'


def create_cloud_index(
    bucket_url=f"s3://{S3_BUCKET}",
    storage_options={},
    logger=None,
):
    if logger is None:
        from loguru import logger

    fsmap = fsspec.get_mapper(bucket_url, **storage_options)
    index_dict = {'plots': [], 'hitl_notes': []}

    for item in fsmap.keys():
        if item.endswith('.png'):
            index_dict['plots'].append(item)
        elif item.startswith('HITL_notes') and item.endswith('.csv'):
            index_dict['hitl_notes'].append(item)
        elif item == INDEX_FILE:
            logger.info(f"{INDEX_FILE} file... skipping.")
        else:
            logger.warning(f"Unknown file found: {item}.")

    with fsmap.fs.open(f"{fsmap.root}/{INDEX_FILE}", mode='w') as f:
        json.dump(index_dict, f)


def create_local_index():
    plots_json = PLOTSDIR.joinpath(INDEX_FILE)
    hitl_json = HITLDIR.joinpath(INDEX_FILE)
    plotsmapper = fsspec.get_mapper(str(PLOTSDIR))
    hitlmapper = fsspec.get_mapper(str(HITLDIR))
    plots_index = [
        item for item in plotsmapper.keys() if item.endswith('.png')
    ]
    hitl_index = [item for item in hitlmapper.keys() if item.endswith('.csv')]

    plots_json.write_text(json.dumps(plots_index))
    hitl_json.write_text(json.dumps(hitl_index))


def parse_args():
    arg_parser = argparse.ArgumentParser(description='QAQC Index Creator')

    arg_parser.add_argument('--cloud', action="store_true")

    return arg_parser.parse_args()


def main():
    from loguru import logger

    args = parse_args()
    if args.cloud is True:
        create_cloud_index(logger=logger)
    else:
        create_local_index()
