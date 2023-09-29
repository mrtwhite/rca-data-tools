FROM prefecthq/prefect:2-python3.10

COPY ./ /tmp/rca_data_tools

RUN pip install prefect-aws
RUN pip install -e /tmp/rca_data_tools
