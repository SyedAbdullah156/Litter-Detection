"""
Data Training Script
Uses: YOLO for training model
"""

import mlflow
import mlflow.pytorch
from ultralytics import YOLO
import torch
from datetime import datetime

class LitterDetectionTrainer:
    """Train YOLOv8 model with MLflow tracking"""
    
    def __init__(self, data_yaml, model_name='yolov8n.pt', experiment_name='litter-detection'):
        self.data_yaml = data_yaml
        self.model_name = model_name
        self.experiment_name = experiment_name
        
        # Setup MLflow
        mlflow.set_experiment(experiment_name)
        
    def train(self, epochs=100, img_size=640, batch_size=16, optimizer='SGD', lr=0.01, device='0', project='runs/detect', name='litter_detection'):
        """Train the model with MLflow tracking"""
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            params = {
                'model': self.model_name,
                'epochs': epochs,
                'img_size': img_size,
                'batch_size': batch_size,
                'device': device,
                'optimizer': optimizer,
                'lr0': lr
            }
            mlflow.log_params(params)
            
            # Load model
            model = YOLO(self.model_name)
            
            # Train
            print("🚀 Starting training...")
            results = model.train(
                data=self.data_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                project=project,
                name=name,
                exist_ok=True,
                pretrained=True,
                optimizer=optimizer,
                verbose=True,
                patience=50,
                save=True,
                plots=True,
                # Data augmentation
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.0,
                copy_paste=0.0
            )
            
            # Log metrics
            metrics = {
                'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'precision': results.results_dict.get('metrics/precision(B)', 0),
                'recall': results.results_dict.get('metrics/recall(B)', 0),
            }
            mlflow.log_metrics(metrics)
            
            # Log model
            best_model_path = 'runs/detect/litter_detection/weights/best.pt'
            mlflow.log_artifact(best_model_path, artifact_path="model")
            
            # Log training plots
            plots_dir = 'runs/detect/litter_detection'
            for plot_file in Path(plots_dir).glob('*.png'):
                mlflow.log_artifact(str(plot_file), artifact_path="plots")
            
            print("✅ Training completed and logged to MLflow!")
            
            return model, results