from ultralytics import YOLO
import argparse
import os
import torch

def train_yolo(data_path, epochs=50, imgsz=640, batch=4):
    """
    Trains a YOLOv8 model for defect localization.
    Optimized for GTX 1060 6GB with memory management.
    """
    # Enable CUDA memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load a pretrained model (yolov8n.pt is the smallest/fastest)
    model = YOLO('yolov8n.pt') 

    # Train with optimized settings for 6GB GPU
    # Using gradient accumulation (effective batch = batch * 2)
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='texvision_yolo_train',
        project='runs/train',
        exist_ok=True,
        patience=20,  # Early stopping
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        cache=False,  # Disable caching to save memory
        workers=2,  # Reduce workers to save memory
        amp=True,  # Automatic Mixed Precision
        close_mosaic=10  # Disable mosaic augmentation in last 10 epochs
    )
    
    print(f"✓ YOLO Training complete. Results saved to {results.save_dir}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/TILDA_yolo/data.yaml', help='path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    
    args = parser.parse_args()
    
    train_yolo(
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
