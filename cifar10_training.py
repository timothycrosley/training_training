"""
CIFAR-10 Training Pipeline - ML Platform Perspective
====================================================

This implementation demonstrates key architectural patterns and pain points
that ML engineers face, highlighting opportunities for platform solutions.

Key Platform Challenges Addressed:
1. Experiment tracking and reproducibility
2. Resource management and monitoring
3. Model versioning and comparison
4. Data pipeline optimization
5. Collaborative development workflows
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import time
from pathlib import Path
import hashlib


# =============================================================================
# 1. CONFIGURATION MANAGEMENT
# Platform Pain Point: Users need reproducible, shareable configs
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration class that makes experiments reproducible.
    
    Platform Opportunity: Config versioning, templates, parameter sweeps
    """
    # Data settings
    batch_size: int = 128
    num_workers: int = 4
    data_augmentation: bool = True
    
    # Model architecture
    model_name: str = "ResNetCustom"
    num_classes: int = 10
    dropout_rate: float = 0.1
    
    # Training hyperparameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 7
    
    # Optimization
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    gradient_clip_norm: float = 1.0
    
    # Platform integration
    experiment_name: str = "cifar10_baseline"
    save_checkpoints: bool = True
    log_frequency: int = 100
    
    def to_dict(self) -> Dict:
        """Serialize config for logging/storage"""
        return {k: v for k, v in self.__dict__.items()}
    
    def get_hash(self) -> str:
        """Generate unique hash for this configuration"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


# =============================================================================
# 2. DATA PIPELINE WITH MONITORING
# Platform Pain Point: Data versioning, lineage, pipeline optimization
# =============================================================================

class DataPipeline:
    """Handles data loading with monitoring hooks for platform integration.
    
    Platform Opportunities:
    - Data versioning and lineage tracking
    - Pipeline performance monitoring
    - Automatic data validation
    - Cross-team dataset sharing
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_stats = {}
        
    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Define data augmentation pipeline.
        
        Platform Note: Users often want to A/B test different augmentation strategies
        """
        if self.config.data_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        return train_transform, val_transform
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test dataloaders with monitoring.
        
        Platform Integration: This is where you'd hook into:
        - Data lineage tracking
        - Performance monitoring (loading times, memory usage)
        - Data quality validation
        """
        train_transform, val_transform = self.get_transforms()
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        
        # Split training into train/val
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=val_transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True, 
            num_workers=self.config.num_workers,
            pin_memory=True  # Important for GPU performance
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size,
            shuffle=False, 
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size,
            shuffle=False, 
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Store dataset statistics for platform monitoring
        self.data_stats = {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'num_classes': 10,
            'input_shape': (3, 32, 32)
        }
        
        return train_loader, val_loader, test_loader


# =============================================================================
# 3. MODEL ARCHITECTURE WITH MODERN PATTERNS
# Platform Pain Point: Architecture experimentation and sharing
# =============================================================================

class BasicBlock(nn.Module):
    """Residual block - fundamental building pattern in modern CNNs"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection for residual learning
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection (key innovation of ResNets)
        out += self.skip_connection(residual)
        out = torch.relu(out)
        
        return out


class ResNetCustom(nn.Module):
    """Custom ResNet architecture demonstrating modern design patterns.
    
    Platform Opportunity: Architecture templates, automated architecture search
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.1):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers (progressively increase channels, decrease spatial size)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling (more parameter efficient than fully connected)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights properly (important for training stability)
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction backbone
        x = torch.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification head
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# =============================================================================
# 4. TRAINING ORCHESTRATION WITH MONITORING
# Platform Pain Point: Experiment tracking, resource management, reproducibility
# =============================================================================

class MetricsTracker:
    """Tracks training metrics for platform integration.
    
    Platform Opportunity: Real-time dashboards, automatic alerting, comparison tools
    """
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'epoch_times': [],
            'gpu_memory': []
        }
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_latest(self, key: str):
        return self.metrics[key][-1] if self.metrics[key] else None
    
    def save(self, path: Path):
        """Save metrics for platform storage"""
        with open(path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)


class Trainer:
    """Main training orchestrator with platform integration hooks.
    
    This class demonstrates the complexity ML engineers face and opportunities
    for platform simplification.
    """
    
    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device
        self.metrics = MetricsTracker()
        
        # Create experiment directory (platform would manage this)
        self.experiment_dir = Path(f'experiments/{config.experiment_name}_{config.get_hash()}')
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration for reproducibility
        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    def setup_model_and_training(self, train_loader: DataLoader):
        """Initialize model, optimizer, scheduler, loss function"""
        
        # Model initialization
        self.model = ResNetCustom(
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer setup (platform could provide optimizer recommendations)
        if self.config.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.config.optimizer} not supported")
        
        # Learning rate scheduler
        if self.config.scheduler == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs
            )
        
        # Print model info (platform would show this in dashboard)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with detailed monitoring"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Platform Integration: Real-time logging
            if batch_idx % self.config.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                batch_acc = 100. * correct / total
                print(f'Batch {batch_idx}/{len(train_loader)}: '
                      f'Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%, LR: {current_lr:.6f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model performance"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint (platform would manage versioning)"""
        if not self.config.save_checkpoints:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': val_acc,
            'config': self.config.to_dict()
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.experiment_dir / 'checkpoint_latest.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.experiment_dir / 'checkpoint_best.pth')
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop with early stopping and monitoring"""
        print(f"Starting training: {self.config.experiment_name}")
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Device: {self.device}")
        
        self.setup_model_and_training(train_loader)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Track metrics
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.metrics.update(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                learning_rates=current_lr,
                epoch_times=epoch_time,
                gpu_memory=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )
            
            # Progress reporting (platform would show in real-time dashboard)
            print(f'Epoch {epoch+1}/{self.config.epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  LR: {current_lr:.6f}, Time: {epoch_time:.1f}s')
            
            # Early stopping logic
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                patience_counter = 0
                print(f'  *** New best validation accuracy: {val_acc:.2f}% ***')
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
            
            print('-' * 60)
        
        # Save final metrics
        self.metrics.save(self.experiment_dir)
        print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
        return best_val_acc


# =============================================================================
# 5. EVALUATION AND ANALYSIS
# Platform Pain Point: Model comparison and analysis tools
# =============================================================================

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict:
    """Comprehensive model evaluation with detailed metrics.
    
    Platform Opportunity: Automated model comparison, performance regression detection
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate comprehensive metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    test_acc = 100. * np.mean(np.array(all_predictions) == np.array(all_targets))
    test_loss /= len(test_loader)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Detailed classification report
    report = classification_report(all_targets, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(all_targets, all_predictions).tolist()
    }
    
    return results


# =============================================================================
# 6. MAIN EXECUTION WITH PLATFORM INTEGRATION POINTS
# =============================================================================

def main():
    """Main execution function demonstrating full ML workflow.
    
    Platform Integration Points:
    - Resource allocation and monitoring
    - Experiment tracking and comparison
    - Automated hyperparameter tuning
    - Model registry and deployment
    """
    
    # Set random seeds for reproducibility (platform responsibility)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device setup (platform would handle resource allocation)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration (platform would provide UI for this)
    config = TrainingConfig(
        experiment_name="cifar10_resnet_demo",
        batch_size=128,
        learning_rate=0.001,
        epochs=20,  # Reduced for demo
        early_stopping_patience=5
    )
    
    print("Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    print(f"Config Hash: {config.get_hash()}")
    
    # Data pipeline setup
    data_pipeline = DataPipeline(config)
    train_loader, val_loader, test_loader = data_pipeline.create_dataloaders()
    
    print(f"\nDataset Statistics:")
    print(json.dumps(data_pipeline.data_stats, indent=2))
    
    # Training
    trainer = Trainer(config, device)
    best_val_acc = trainer.train(train_loader, val_loader)
    
    # Load best model for final evaluation
    checkpoint = torch.load(trainer.experiment_dir / 'checkpoint_best.pth')
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    results = evaluate_model(trainer.model, test_loader, device)
    
    print(f"Final Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"Final Test Loss: {results['test_loss']:.4f}")
    
    # Save results (platform would store in model registry)
    with open(trainer.experiment_dir / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nExperiment artifacts saved to: {trainer.experiment_dir}")
    
    # Platform Integration: This is where you'd:
    # - Register the model in model registry
    # - Compare against baseline models
    # - Trigger deployment pipeline if metrics are good
    # - Send notifications to team
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
    
    print("\n" + "="*60)
    print("PLATFORM INTEGRATION OPPORTUNITIES")
    print("="*60)
    print("""
    Based on this workflow, your ML platform could provide:
    
    1. EXPERIMENT MANAGEMENT
       - Automatic config versioning and comparison
       - Real-time training dashboards
       - Experiment comparison tools
       
    2. RESOURCE ORCHESTRATION  
       - GPU allocation and queuing
       - Distributed training coordination
       - Cost optimization
       
    3. DATA MANAGEMENT
       - Dataset versioning and lineage
       - Data quality monitoring
       - Pipeline performance tracking
       
    4. MODEL LIFECYCLE
       - Model registry with versioning
       - Automated testing and validation
       - Deployment pipeline integration
       
    5. COLLABORATION
       - Shared experiment spaces
       - Code and artifact sharing
       - Team notifications and reporting
    """)
