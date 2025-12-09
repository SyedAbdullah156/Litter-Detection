"""
Data Training Script
Uses: YOLO for training model
"""

import mlflow
import mlflow.pytorch
from ultralytics import YOLO
import torch
from datetime import datetime
from pathlib import Path

class LitterDetectionTrainer:
    """Train YOLOv8 model with MLflow tracking"""
    
    def __init__(self, data_yaml, model_name, experiment_name):
        self.data_yaml = data_yaml
        self.model_name = model_name
        self.experiment_name = experiment_name
        
        # Setup MLflow
        mlflow.set_experiment(experiment_name)
        
    def train(self, epochs=100, img_size=640, batch_size=16, device='cuda', project='runs/detect', name='litter_detection', optimizer='SGD', patience=50, lr=0.01, fraction=0.1, augmentation=None, tags=None):
        """Train the model with MLflow tracking"""
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

            # Applying Tags to Run
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, v)
            
            # Auto-logging will log all things necessary
            mlflow.autolog()
            
            # Load model
            model = YOLO(self.model_name)
            
            # Train
            print("🚀 Starting training...")

            # Checking for augmentation
            if augmentation:
                results = model.train(
                    data=self.data_yaml,
                    epochs=epochs,
                    imgsz=img_size,
                    batch=batch_size,
                    device=device,
                    project=project,
                    name=name,
                    optimizer=optimizer,
                    patience=patience,
                    lr0=lr,
                    fraction=fraction,
                    exist_ok=True,
                    pretrained=True,
                    verbose=True,
                    save=True,
                    plots=True,
                    **augmentation,
                )
            else:
                results = model.train(
                    data=self.data_yaml,
                    epochs=epochs,
                    imgsz=img_size,
                    batch=batch_size,
                    device=device,
                    project=project,
                    name=name,
                    optimizer=optimizer,
                    patience=patience,
                    lr0=lr,
                    fraction=fraction,
                    exist_ok=True,
                    pretrained=True,
                    verbose=True,
                    save=True,
                    plots=True,
                )

            print("✅ Training completed and logged to MLflow!")
            mlflow.end_run()

            return model, results
        
if __name__ == "__main__":
    trainer = LitterDetectionTrainer(
        data_yaml='../config/data_config.yaml' ,
        model_name='yolov8n.pt',  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
        experiment_name='litter-detection-taco'
    )

    model, results = trainer.train(
        epochs=100,
        img_size=640,
        batch_size=16,
        device='0'  # Use '0' for GPU, 'cpu' for CPU
    )