import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
import os
import sys
from tqdm import tqdm
import torchvision.models as models
from torchvision import datasets, transforms
import time
import multiprocessing

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.loss import get_loss
from evaluation.metrics import MetricCalculator

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_model(config, num_classes):
    model_name = config['model']['name']
    pretrained = config['model']['pretrained']
    
    if 'resnet' in model_name:
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif 'efficientnet' in model_name:
        model = models.efficientnet_b0(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    return model

def train(args):
    # Enable CUDA memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA Memory Cache Cleared. Device: {torch.cuda.get_device_name(0)}")

    # Load Configs
    train_cfg = load_config(args.config)
    dataset_cfg = load_config('configs/datasets.yaml')
    model_cfg = load_config('configs/model.yaml')
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Simple Dataset Loading (Placeholder for custom Dataset classes if needed)
    # Using ImageFolder for DTD standard structure
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Determine dataset to use (default to dtd if not specified in training config)
    dataset_name = train_cfg['training'].get('dataset_name', 'dtd')
    if dataset_name not in dataset_cfg['datasets']:
        raise ValueError(f"Dataset {dataset_name} not found in dataset config")
        
    print(f"Training on dataset: {dataset_name}")
    data_dir = dataset_cfg['datasets'][dataset_name]['root_dir']
    
    # Ensure data exists (mocking for script completeness if data not present)
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} not found. Creating dummy for test.")
        os.makedirs(os.path.join(data_dir, 'train', 'class1'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'val', 'class1'), exist_ok=True)
    
    try:
        train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_val)
    except FileNotFoundError:
        print("Dataset structure incomplete. using dummy data path for now")
        train_dataset = datasets.FakeData(transform=transform_train)
        val_dataset = datasets.FakeData(transform=transform_val)

    # Optimized DataLoader: Use CPU workers to fetch data while GPU trains
    # Prioritize dataset config for num_workers, fallback to dynamic
    dataset_workers = dataset_cfg['datasets'][dataset_name].get('num_workers', 4)
    # Consider CPU count limit
    num_workers = min(dataset_workers, multiprocessing.cpu_count())
    
    print(f"DataLoader Config: num_workers={num_workers} (Config requested: {dataset_workers}, CPU count: {multiprocessing.cpu_count()})")
    print(f"Pin Memory: {True if device.type == 'cuda' else False}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg['training']['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False # Critical for CPU->GPU transfer speed
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_cfg['training']['batch_size'], 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Model
    num_classes = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else 10
    model = get_model(model_cfg, num_classes)
    model = model.to(device)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg['training']['optimizer']['lr'])
    criterion = get_loss(train_cfg['training']['loss'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg['training']['num_epochs'])
    
    # Logging
    writer = SummaryWriter(log_dir=train_cfg['training']['logging']['log_dir'])
    metric_calc = MetricCalculator()
    
    best_acc = 0.0
    global_step = 0
    
    for epoch in range(train_cfg['training']['num_epochs']):
        # Training
        model.train()
        running_loss = 0.0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['training']['num_epochs']}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, col_preds = torch.max(outputs, 1)
            train_preds.extend(col_preds.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            
            global_step += 1
            
            # Log every few batches so User sees TensorBoard updates IMMEDIATELY
            if global_step % 10 == 0:
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            
            pbar.set_postfix({'loss': loss.item()})
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, col_preds = torch.max(outputs, 1)
                val_preds.extend(col_preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Metrics
        train_metrics = metric_calc.compute(train_targets, train_preds)
        val_metrics = metric_calc.compute(val_targets, val_preds)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        
        print(f"Epoch {epoch+1}: Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Checkpoint
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            save_path = os.path.join(train_cfg['training']['checkpointing']['save_dir'], 'best_model.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training.yaml', help='Path to training config')
    args = parser.parse_args()
    train(args)
