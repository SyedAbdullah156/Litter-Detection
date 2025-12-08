"""
Classification Training with EfficientNet
Uses: PyTorch, CUDA, MLflow, EfficientNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from configs.mlflow_config import setup_mlflow, log_model_info
import time

class LitterClassifier:
    def __init__(self, num_classes=6, model_name='efficientnet-b0'):
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Initialize model
        self.model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        self.model = self.model.to(self.device)
    
    def prepare_data(self, data_dir, batch_size=32, num_workers=4):
        """Prepare data loaders"""
        
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # No augmentation for validation
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(
            root=f"{data_dir}/train",
            transform=train_transform
        )
        
        val_dataset = datasets.ImageFolder(
            root=f"{data_dir}/val",
            transform=val_transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.classes = train_dataset.classes
        
        print(f"✅ Train batches: {len(self.train_loader)}")
        print(f"✅ Val batches: {len(self.val_loader)}")
        print(f"✅ Classes: {self.classes}")
        
        return train_dataset, val_dataset
    
    def train(self, num_epochs=50, learning_rate=0.001, save_dir='./models'):
        """
        Training loop with MLflow tracking
        """
        
        # Setup MLflow
        mlflow_client = setup_mlflow("litter-classification")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"efficientnet_{time.strftime('%Y%m%d_%H%M%S')}"):
            
            # Log hyperparameters
            mlflow.log_param("model", self.model_name)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("batch_size", self.train_loader.batch_size)
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("device", str(self.device))
            
            # Log model info
            log_model_info(self.model, self.model_name)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=5, factor=0.5
            )
            
            best_val_acc = 0.0
            
            # Training loop
            for epoch in range(num_epochs):
                print(f"\n{'='*60}")
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"{'='*60}")
                
                # TRAIN
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                train_pbar = tqdm(self.train_loader, desc="Training")
                for images, labels in train_pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    train_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100 * train_correct / train_total:.2f}%'
                    })
                
                train_loss = train_loss / len(self.train_loader)
                train_acc = 100 * train_correct / train_total
                
                # VALIDATION
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    val_pbar = tqdm(self.val_loader, desc="Validation")
                    for images, labels in val_pbar:
                        images, labels = images.to(self.device), labels.to(self.device)
                        
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                        val_pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'acc': f'{100 * val_correct / val_total:.2f}%'
                        })
                
                val_loss = val_loss / len(self.val_loader)
                val_acc = 100 * val_correct / val_total
                
                # Learning rate scheduling
                scheduler.step(val_acc)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Log metrics to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)
                
                # Print summary
                print(f"\n📊 Epoch {epoch+1} Summary:")
                print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"   Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
                print(f"   Learning Rate: {current_lr:.6f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_path = f"{save_dir}/efficientnet_best.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'classes': self.classes
                    }, model_path)
                    
                    # Log model to MLflow
                    mlflow.pytorch.log_model(self.model, "model")
                    
                    print(f"   ✅ New best model saved! Val Acc: {val_acc:.2f}%")
            
            print(f"\n🎉 Training Complete!")
            print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
            
            # Log best accuracy
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            
            return best_val_acc

if __name__ == "__main__":
    # Initialize classifier
    classifier = LitterClassifier(num_classes=6, model_name='efficientnet-b0')
    
    # Prepare data
    classifier.prepare_data(
        data_dir="./data/processed",
        batch_size=32,
        num_workers=4
    )
    
    # Train
    classifier.train(
        num_epochs=50,
        learning_rate=0.001,
        save_dir="./models"
    )