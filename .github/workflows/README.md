# Github workflows

This directory contains all of the github workflow configuration for Github Actions.

- **main.yaml**: CI workflow that installs the package and runs the test with `pytest`.
- **register_pipeline.yaml**: Pipeline registration workflow that register the qaqc dashboard prefect flow to prefect cloud. (This workflow runs on push to `main` branch).

For documentation on github workflow syntax, see [official docs](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions).
