# QAQC Dashboard

The qaqc dashboard is a web application serving plots of the data to perform quality control.

The application can be found at [qaqc.ooi-rca.net](https://qaqc.ooi-rca.net/).

The QAQC dashboard is hosted on Amazon Web Services (AWS) in an S3 Bucket, which is then served via a Content Delivery Network (CDN). The plots found in the application is created by python scripts that have been made into data pipelines running on AWS via [Prefect 1.0](https://docs-v1.prefect.io/) data workflow and orchestration tools.

The code for the infrastructure, backend, and frontend are hosted in 3 separate repositories: `rca-data-tools`, `QAQC_dashboard`, and `cloud-infrastructure` (*private*) within the `OOI-CabledArray` organization.

- [`rca-data-tools`](https://github.com/OOI-CabledArray/rca-data-tools): contains the majority of the code to perform the creation of png plots and csv plots for the dashboard.
- [`QAQC_dashboard`](https://github.com/OOI-CabledArray/QAQC_dashboard) contains the frontend code for the dashboard.
- [`cloud-infrastructure`](https://github.com/OOI-CabledArray/cloud-infrastructure): contains terraform code for deploying the underlying infrastructure such as Virtual Private Cloud (VPC), CDN, Elastic Container Service (ECS) Tasks, Identity Access Management (IAM), and S3 Buckets.

## Data tools ([**rca-data-tools**](https://github.com/OOI-CabledArray/rca-data-tools))

This is a package full of data tools for RCA team. The current main tool is contained within the `qaqc` module, but can expand for other use cases.

### qaqc

This module can be found at `./rca_data_tools/qaqc`. This contains code for creating prefect flows, which registrations are run within github actions, the configuration can be found at [`.github/workflows/register_pipeline.yaml`](../../.github/workflows/register_pipeline.yaml).

## QAQC Dashboard ([**QAQC_dashboard**](https://github.com/OOI-CabledArray/QAQC_dashboard))

This is a repository that contains the github actions configurations to run the plot, index, and HITL csvs creations. Additionally, it also deploys the client application that are hosted in AWS built with vue.js.

### Github actions

All of the various configurations can be found at the [github workflows](https://github.com/OOI-CabledArray/QAQC_dashboard/tree/main/.github/workflows) directory.

## Cloud Infrastructure ([**cloud-infrastructure**](https://github.com/OOI-CabledArray/cloud-infrastructure))

This is the repository for the cloud components needed to run and host the QAQC Dashboard. It is currently private, so access to repository is needed. Please ask [Wendi Ruef](https://github.com/wruef) for access.

This page serves as the pointers to each of the repositories, please refer to each repository for more detailed documentation.
