import torch
import yaml
import argparse
import os
import sys
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import MetricCalculator
from evaluation.confusion_matrix import plot_confusion_matrix

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def validate(args):
    model_cfg = load_config('configs/model.yaml')
    dataset_cfg = load_config('configs/datasets.yaml')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data (Similar logic to train.py)
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset_name = args.dataset if args.dataset else 'dtd'
    print(f"Validating on dataset: {dataset_name}")
    data_dir = dataset_cfg['datasets'][dataset_name]['root_dir']
    # If data doesn't exist, use FakeData for demonstration
    if os.path.exists(os.path.join(data_dir, 'test')):
        val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform_val)
    else:
        print("Test data not found, using FakeData.")
        val_dataset = datasets.FakeData(transform=transform_val)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Load Model
    num_classes = len(val_dataset.classes) if hasattr(val_dataset, 'classes') else 10
    
    # Reconstruct model (assuming ResNet50 for now based on config defaults)
    model_name = model_cfg['model']['name']
    if 'resnet50' in model_name:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    metric_calc = MetricCalculator()
    metrics = metric_calc.compute(all_targets, all_preds)
    
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Plot Confusion Matrix
    if hasattr(val_dataset, 'classes'):
        plot_confusion_matrix(all_targets, all_preds, classes=val_dataset.classes, save_path='logs/confusion_matrix.png')
        print("Confusion matrix saved to logs/confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='dtd', help='Dataset name (dtd, tilda)')
    args = parser.parse_args()
    validate(args)
