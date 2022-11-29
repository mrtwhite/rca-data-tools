# Dockerfile for building prefect pipeline
# this file will be used when the pipeline image is being built
# it is based on the `pangeo-notebook` image,
# which contains a lot of data science packages

FROM pangeo/pangeo-notebook
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y tzdata
USER ${NB_USER}
RUN mamba install -n notebook -c conda-forge --yes \
      fastparquet \
      python=3.8 \
      "prefect<1.0" \
      "pycurl<7.45" \
      "setuptools<59" \
      "wheel>=0.29.0"
ENV PATH=/srv/conda/envs/notebook/bin:$PATH
ENV JUPYTERHUB_USER=rca-qaqc
# List all the packages in notebook env
RUN conda list -n notebook
