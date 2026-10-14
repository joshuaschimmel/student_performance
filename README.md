# Student Performance

This project is estimating student performance using Azure Machine Learning with MLOps.
The goal is to have a model endpoint online that can predict a students performance based on input data.
The ML model should be limited to simple multiple linear regression.

This needs:

* provisioning of azure resources with infra script
* get and store data in azure
* training and validating model in azure
* testing different models
* HPO
* deploying model live
* notebook example utilizing endpoint
* github actions workflow retraining model if model config, experiment or data changes

## Data

Synthetic dataset is from Kaggle: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/data

Run `bash download_dataset` to download and unzip data into `data/`.

The dataset is designed for multiple linear regression.

TODO describe dataset

## Infra

use azure cli to provision resources

## Environments

For this example project only a dev environment, nothing live for the public.
Results should be documented in a notebook.

## Notes and thinkings

* how to submit jobs?
* jobs should run in cluster
* use command jobs or automl?
* should be triggerable from GitHub Actions
* 
