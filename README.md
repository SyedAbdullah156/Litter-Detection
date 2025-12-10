# Litter Detection

This repository contains all code, experiments, and artifacts related to my litter-detection project using the TACO dataset and YOLO-based models.

## Kaggle Notebooks
I also used Kaggle notebooks during development. They can be accessed here:

- https://www.kaggle.com/code/syedabdullah1247/official-taco-dataset  
- https://www.kaggle.com/code/syedabdullah1247/taco-dataset-yolo-format/

## Project Report
Full project report (Overleaf):  
https://www.overleaf.com/read/wkhgswgcbwny#bc0564

## Plagiarism Report
The official plagiarism report from the university library is pending due to processing delays.  
I have temporarily uploaded an automated report generated using https://paperpal.com/.  
The official report will be added here once received.

## Inference Outputs
All inference results on images and videos are stored using DVC and are available on DagsHub.  
This includes predictions, evaluation runs, and processed outputs.

## Model Registry
All trained models are logged and versioned in MLflow (via DagsHub).  
Different versions can be browsed, compared, and downloaded directly from the MLflow dashboard.

## Experiments to Consider
Only the following experiment groups contain valid models, metrics, and parameters:

1. `litter-detection-taco`
2. `litter-detection-taco-on-modified-data`

All other experiment runs were exploratory and should not be used for evaluation.

## Notebook Execution
Most runs were executed locally on my laptop.  
One additional run was executed on Kaggle for GPU availability, marked in the source as `colab_notebook.py`.
