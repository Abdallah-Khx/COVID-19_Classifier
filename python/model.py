import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision import models
import mlflow
import mlflow.pytorch
import numpy as np

class COVIDClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(COVIDClassifier, self).__init__()
        # Load pretrained ResNet50
        self.base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Modify the final layers
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

class ModelTrainer:
    def __init__(self, model, device, learning_rate=0.0001):
        self.model = model
        self.device = device
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.2, patience=3, verbose=True
        )

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        return epoch_loss, accuracy

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        return val_loss, accuracy

    def train(self, train_loader, val_loader, epochs=25):
        # Do NOT start a new run here! Only log params and metrics.
        mlflow.log_params({
            "num_classes": self.model.base_model.fc[-1].out_features,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "epochs": epochs,
            "base_model": "ResNet50"
        })
        
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }, step=epoch)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                mlflow.pytorch.log_model(self.model, "model")
                
        example_input = np.random.randn(1, 3, 299, 299).astype(np.float32)  # Use numpy array
        mlflow.pytorch.log_model(self.model, "model", input_example=example_input)
                
        torch.save(self.model.state_dict(), "covid_classifier.pth")
                
        # Unfreeze last block
        for name, param in self.model.base_model.named_parameters():
            if "layer4" in name:  # Unfreeze last ResNet block
                param.requires_grad = True
                
        return self.model 