from ultralytics import YOLO
import argparse
import os
import torch

def train_optimized(data_path, epochs=150, imgsz=640, batch=4):
    """
    Trains a YOLOv8 model with optimized settings for higher accuracy.
    Uses transfer learning from previous best weights.
    """
    # 1. Enable CUDA memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 2. Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🚀 Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 3. Load Previous Best Weights (Transfer Learning)
    # Adjust path if needed based on your file structure
    base_weights = 'runs/detect/runs/train/texvision_yolo_train/weights/best.pt'
    if os.path.exists(base_weights):
        print(f"✅ Loading pretrained weights from: {base_weights}")
        model = YOLO(base_weights)
    else:
        print(f"⚠️  Pretrained weights not found at {base_weights}. Starting from yolov8n.pt")
        model = YOLO('yolov8n.pt')

    # 4. Train with Aggressive Augmentations & Optimized Hyperparameters
    print("🔥 Starting Optimized Training...")
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=os.path.abspath('runs/train/optimized_run'),
        name='texvision_yolo_optimized_v2',
        exist_ok=True,
        
        # Optimization Strategy
        patience=50,        # Extended patience to prevent early stopping
        save=True,
        save_period=5,      # Save more frequently
        cos_lr=True,        # Cosine annealing learning rate
        optimizer='auto',   # Let YOLO decide (usually SGD) for stability
        lr0=0.01,           # Default for SGD (if AdamW, use 0.001)
        lrf=0.01,           # Final LR = lr0 * lrf
        
        # Augmentations (Aggressive for Recall)
        mixup=0.1,          # Reduced slightly to prevent confusion
        copy_paste=0.15,    # Reduced to 0.15 to improve Precision
        mosaic=1.0,         # Keep mosaic on
        close_mosaic=10,    # Turn off mosaic for last 10 epochs for stability
        degrees=10.0,       # Slight rotation
        fliplr=0.5,         # Left-right flip
        
        # System Limits
        cache=False,        # Disable RAM cache
        workers=2,
        amp=True
    )
    
    print(f"🎉 Training Complete! Best weights: {results.save_dir}/weights/best.pt")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/TILDA_yolo/data.yaml')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    
    train_optimized(args.data, epochs=args.epochs, batch=args.batch)
