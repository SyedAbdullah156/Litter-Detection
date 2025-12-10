# Litter Detection

This repository contains all code, experiments, datasets, and artifacts used for my litter-detection project based on the TACO dataset and YOLO object detection models. The workflow combines dataset preprocessing, model training, experiment tracking, and inference pipelines.

## Technologies and Tools Used

### Machine Learning & Computer Vision
- **YOLOv8** (Ultralytics) for training and inference  
- **COCO Format** for dataset annotations  
- **OpenCV (cv2)** for image/video preprocessing and visualization  
- **PyTorch** backend used automatically through YOLOv8  

### Experiment Tracking & Versioning
- **MLflow** (via DagsHub) for experiment tracking, metrics logging, parameters, and model registry  
- **DVC (Data Version Control)** for dataset and inference output versioning  
- **DagsHub** as the central hub connecting Git, DVC, and MLflow  

### Platforms Used
- **Kaggle** notebooks for faster GPU-based training  
- **Local Machine** for development and most training runs  
- **Overleaf** for writing and managing the final project report  

---

## Kaggle Notebooks
Project development also used Kaggle notebooks:

- https://www.kaggle.com/code/syedabdullah1247/official-taco-dataset  
- https://www.kaggle.com/code/syedabdullah1247/taco-dataset-yolo-format/

## Project Report
Full Overleaf project report:  
https://www.overleaf.com/read/wkhgswgcbwny#bc0564

## Plagiarism Report
The official plagiarism report from the university library is pending.  
A temporary automated report from https://paperpal.com/ has been uploaded and will be replaced once the official report is issued.

## Inference Outputs
All inference videos, images, predictions, and processed data are tracked using **DVC** and stored on **DagsHub**. These can be accessed directly through the repository.

## Model Registry
All trained YOLO models are automatically registered and versioned through **MLflow**.  
You can browse, compare, and download different versions directly from the MLflow UI on DagsHub.

## Experiments to Consider
Only the following experiment groups contain complete and valid models, metrics, and parameters:

1. `litter-detection-taco`  ( This contains models trained on original TACO dataset )
2. `litter-detection-taco-on-modified-data` ( This contains models trained on modified TACO dataset )

All other experiments are exploratory and should not be used for evaluation.

## Notebook Execution
Most training runs were executed on my local machine.  
One additional run was executed on Kaggle for GPU access and is labeled in the source as `colab_notebook.py`.
