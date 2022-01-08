import json
from pathlib import Path
import fsspec

HERE = Path(__file__).absolute().parent.parent.parent
PLOTSDIR = HERE.joinpath('QAQCplots')
HITLDIR = HERE.joinpath('HITL_notes')
INDEX_FILE = 'index.json'


def main():
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


if __name__ == "__main__":
    main()
