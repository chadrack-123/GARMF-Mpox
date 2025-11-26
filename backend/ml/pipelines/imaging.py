"""
Imaging ML pipelines using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Custom dataset for image classification"""
    
    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ImagingPipeline:
    """Pipeline for image classification tasks"""
    
    def __init__(
        self,
        model_name: str = "resnet18",
        image_size: int = 224,
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        """
        Initialize imaging pipeline
        
        Args:
            model_name: Model architecture ('resnet18', 'densenet201', 'vit')
            image_size: Input image size
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            random_state: Random seed
        """
        self.model_name = model_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Set seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    def _get_model(self, num_classes: int) -> nn.Module:
        """Get model architecture"""
        if self.model_name == "resnet18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif self.model_name == "densenet201":
            model = models.densenet201(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif self.model_name == "vit":
            model = models.vit_b_16(pretrained=True)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        return model.to(self.device)
    
    def _get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get train and validation transforms"""
        # ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            normalize
        ])
        
        return train_transform, val_transform
    
    def prepare_data(
        self,
        dataset_dir: Path,
        test_size: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders
        
        Args:
            dataset_dir: Directory with class subdirectories
            test_size: Proportion of test set
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Collect image paths and labels
        image_paths = []
        labels = []
        class_to_idx = {}
        
        class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        for idx, class_dir in enumerate(class_dirs):
            class_to_idx[class_dir.name] = idx
            for img_path in list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")):
                image_paths.append(img_path)
                labels.append(idx)
        
        logger.info(f"Found {len(image_paths)} images in {len(class_to_idx)} classes")
        self.class_to_idx = class_to_idx
        self.num_classes = len(class_to_idx)
        
        # Stratified split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        # Create datasets and loaders
        train_transform, val_transform = self._get_transforms()
        
        train_dataset = ImageDataset(train_paths, train_labels, train_transform)
        val_dataset = ImageDataset(val_paths, val_labels, val_transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Use 0 for Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history
        """
        self.model = self._get_model(self.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
        
        return self.history
    
    def _validate(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # AUC (for binary or multi-class)
        try:
            if self.num_classes == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except:
            auc = None
        
        metrics = {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "auc": float(auc) if auc else None
        }
        
        logger.info(f"Test metrics: {metrics}")
        return metrics
    
    def save_model(self, path: Path):
        """Save trained model"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path, num_classes: int):
        """Load trained model"""
        self.num_classes = num_classes
        self.model = self._get_model(num_classes)
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Model loaded from {path}")
