"""
YOLOv8 Training Script for Fabric Defect Detection
Optimized for Raspberry Pi 4 deployment with small image size and efficient architecture.
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


class YOLOTrainer:
    """YOLOv8 trainer with optimized settings for fabric defect detection"""
    
    def __init__(self, config_path: str):
        """
        Initialize YOLOv8 trainer
        
        Args:
            config_path: Path to training configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.results = None
        
    def setup_model(self):
        """Initialize YOLOv8 model"""
        model_name = self.config.get('model', 'yolov8n.pt')
        print(f"Loading model: {model_name}")
        self.model = YOLO(model_name)
        
    def train(self):
        """
        Train YOLOv8 model with configured parameters
        
        Returns:
            Training results
        """
        if self.model is None:
            self.setup_model()
        
        # Extract training parameters
        data_yaml = self.config.get('data')
        epochs = self.config.get('epochs', 100)
        imgsz = self.config.get('imgsz', 320)
        batch = self.config.get('batch', 16)
        device = self.config.get('device', 0)
        patience = self.config.get('patience', 20)
        workers = self.config.get('workers', 4)
        
        # Additional optimizations
        optimizer = self.config.get('optimizer', 'AdamW')
        lr0 = self.config.get('lr0', 0.001)
        lrf = self.config.get('lrf', 0.01)
        
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"{'='*60}")
        print(f"Model: {self.config.get('model')}")
        print(f"Dataset: {data_yaml}")
        print(f"Image Size: {imgsz}x{imgsz}")
        print(f"Batch Size: {batch}")
        print(f"Epochs: {epochs}")
        print(f"Device: {device}")
        print(f"Patience: {patience}")
        print(f"Optimizer: {optimizer}")
        print(f"Learning Rate: {lr0} -> {lrf}")
        print(f"{'='*60}\n")
        
        # Data augmentation (optimized for fabric textures)
        hsv_h = self.config.get('hsv_h', 0.015)
        hsv_s = self.config.get('hsv_s', 0.7)
        hsv_v = self.config.get('hsv_v', 0.4)
        degrees = self.config.get('degrees', 10.0)
        translate = self.config.get('translate', 0.1)
        scale = self.config.get('scale', 0.5)
        shear = self.config.get('shear', 0.0)
        perspective = self.config.get('perspective', 0.0)
        flipud = self.config.get('flipud', 0.5)
        fliplr = self.config.get('fliplr', 0.5)
        mosaic = self.config.get('mosaic', 1.0)
        mixup = self.config.get('mixup', 0.0)
        
        # Performance and validation
        amp = self.config.get('amp', True)
        save = self.config.get('save', True)
        cache = self.config.get('cache', False)
        
        # Train the model
        self.results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=patience,
            workers=workers,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            mosaic=mosaic,
            mixup=mixup,
            amp=amp,
            save=save,
            save_period=self.config.get('save_period', -1),
            cache=cache,
            val=True,
            plots=True,
            verbose=True,
        )
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Best model saved to: {self.model.trainer.best}")
        print(f"Results saved to: {self.model.trainer.save_dir}")
        print(f"{'='*60}\n")
        
        return self.results
    
    def validate(self):
        """Validate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        print("\nValidating model...")
        metrics = self.model.val()
        
        print(f"\n{'='*60}")
        print(f"Validation Metrics:")
        print(f"{'='*60}")
        print(f"mAP@0.5: {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def export_model(self, format='onnx'):
        """
        Export trained model to different formats
        
        Args:
            format: Export format ('onnx', 'tflite', 'torchscript', etc.)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        print(f"\nExporting model to {format.upper()} format...")
        export_path = self.model.export(format=format)
        print(f"Model exported to: {export_path}")
        
        return export_path


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 for fabric defect detection')
    parser.add_argument('--config', type=str, 
                       default='detection/configs/yolo_training.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation after training')
    parser.add_argument('--export', type=str, default=None,
                       help='Export format (onnx, tflite, etc.)')
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print("Please create the configuration file first.")
        return
    
    # Initialize trainer
    trainer = YOLOTrainer(args.config)
    
    # Train model
    trainer.train()
    
    # Validate if requested
    if args.validate:
        trainer.validate()
    
    # Export if requested
    if args.export:
        trainer.export_model(format=args.export)


if __name__ == "__main__":
    main()
