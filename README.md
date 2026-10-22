# Student Performance

This project is estimating student performance using Azure Machine Learning with MLOps.
The goal is to have a model endpoint online that can predict a students performance based on input data.
The ML model should be limited to simple multiple linear regression.

## Future Work

- ingest and display results from automl run
- use model with parameters and apply HPO
- create endpoint and deploy a model live
- create workflow to deploy a model
- create notebook using inference from notebook
- recreate dataset's mltable config, reading columns as numeric instead of categorical

## Completed Work

- [x] provisioning of azure resources with infra script
- [x] get and store data in azure
- [x] training and validating model in azure
- [x] testing different models
- [x] github actions workflow retraining model if model config, experiment or data changes

## Data

Synthetic dataset is from Kaggle: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/data

Run `bash download_dataset` to download and unzip data into `data/`.

The dataset is designed for multiple linear regression.

## Infra

use azure cli to provision resources

## Environments

For this example project only a dev environment, nothing live for the public.
Results should be documented in a notebook.
