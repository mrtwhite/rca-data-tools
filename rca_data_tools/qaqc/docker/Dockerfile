FROM pangeo/pangeo-notebook
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y tzdata
USER ${NB_USER}
RUN mamba install -n notebook -c conda-forge --yes \
      fastparquet \
      "prefect<1.0" \
      "pycurl<7.45" \
      "setuptools<59" \
      "wheel>=0.29.0"
ENV PATH=/srv/conda/envs/notebook/bin:$PATH
ENV JUPYTERHUB_USER=rca-qaqc
RUN conda info
RUN conda list
