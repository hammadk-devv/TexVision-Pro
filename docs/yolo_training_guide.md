# YOLOv8 Training Guide

Complete guide for training YOLOv8 on fabric defect dataset.

## Prerequisites

- ✅ Annotated dataset (500+ images with bounding boxes)
- ✅ YOLOv8 dependencies installed
- ✅ GPU with CUDA support (recommended)

## Step 1: Verify Dataset

Before training, validate your annotations:

```bash
python scripts/validate_annotations.py --data data/TILDA_yolo/data.yaml --plot
```

**Expected Output:**
- No annotation errors
- Balanced class distribution
- Dataset statistics report

## Step 2: Configure Training

Edit `detection/configs/yolo_training.yaml` if needed:

```yaml
model: yolov8n.pt      # Nano model for RPi4
imgsz: 320             # Image size (320 for RPi4)
epochs: 100            # Training epochs
batch: 16              # Batch size (adjust for GPU)
patience: 20           # Early stopping patience
```

**Key Parameters:**

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `imgsz` | 320 | Optimized for RPi4 |
| `batch` | 16 | Reduce to 8 if OOM |
| `epochs` | 100 | With early stopping |
| `patience` | 20 | Stop if no improvement |

## Step 3: Start Training

### Basic Training

```bash
python detection/yolo_trainer.py --config detection/configs/yolo_training.yaml
```

### With Validation

```bash
python detection/yolo_trainer.py \
    --config detection/configs/yolo_training.yaml \
    --validate
```

### With Export to ONNX

```bash
python detection/yolo_trainer.py \
    --config detection/configs/yolo_training.yaml \
    --validate \
    --export onnx
```

## Step 4: Monitor Training

### TensorBoard

Open TensorBoard to monitor training progress:

```bash
tensorboard --logdir detection/runs/detect/train
```

Access at: `http://localhost:6006`

**Metrics to Watch:**
- `train/box_loss` - Should decrease
- `val/box_loss` - Should decrease
- `metrics/mAP50` - Should increase
- `metrics/mAP50-95` - Should increase

### Console Output

Training will show:
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100     2.84G      1.234      0.567      1.123         64        320
```

## Step 5: Evaluate Results

After training completes, check the results:

```bash
# Results are saved in detection/runs/detect/train/
ls detection/runs/detect/train/weights/
# best.pt  - Best model checkpoint
# last.pt  - Last epoch checkpoint
```

### Validation Metrics

```bash
python detection/yolo_trainer.py \
    --config detection/configs/yolo_training.yaml \
    --validate
```

**Target Metrics:**
- mAP@0.5: > 0.75
- mAP@0.5:0.95: > 0.50
- Precision: > 0.80
- Recall: > 0.75

## Step 6: Test Inference

Test the trained model on sample images:

```bash
python detection/yolo_inference.py \
    --model detection/runs/detect/train/weights/best.pt \
    --source data/TILDA_yolo/test/images/hole_001.jpg \
    --save outputs/test_detection.jpg
```

## Step 7: Export for Deployment

### Export to ONNX (Recommended for RPi4)

```bash
python detection/yolo_trainer.py \
    --config detection/configs/yolo_training.yaml \
    --export onnx
```

Or manually:

```python
from ultralytics import YOLO

model = YOLO('detection/runs/detect/train/weights/best.pt')
model.export(format='onnx')
```

**ONNX Benefits:**
- Faster inference on CPU
- Smaller model size
- Better RPi4 compatibility

## Troubleshooting

### Out of Memory (OOM)

**Symptom:** `CUDA out of memory` error

**Solutions:**
1. Reduce batch size:
   ```yaml
   batch: 8  # or even 4
   ```

2. Reduce image size:
   ```yaml
   imgsz: 256  # instead of 320
   ```

3. Disable cache:
   ```yaml
   cache: false
   ```

### Low mAP

**Symptom:** mAP < 0.70 after training

**Solutions:**
1. **More data**: Annotate more images (aim for 1000+)
2. **Better annotations**: Review and fix annotation errors
3. **Longer training**: Increase epochs or reduce patience
4. **Data augmentation**: Already optimized in config
5. **Larger model**: Try YOLOv8s instead of YOLOv8n

### Overfitting

**Symptom:** Train loss decreases but val loss increases

**Solutions:**
1. **More data**: Add more training images
2. **Data augmentation**: Increase augmentation parameters
3. **Early stopping**: Reduce patience value
4. **Regularization**: Add dropout (requires model modification)

### Slow Training

**Symptom:** Training takes too long

**Solutions:**
1. **Use GPU**: Ensure CUDA is available
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```

2. **Reduce workers**: If CPU bottleneck
   ```yaml
   workers: 2  # instead of 4
   ```

3. **Enable AMP**: Already enabled in config
   ```yaml
   amp: true
   ```

## Hyperparameter Tuning

### For Better Accuracy

```yaml
epochs: 150           # More training
patience: 30          # More patience
lr0: 0.0005          # Lower learning rate
mosaic: 1.0          # Keep mosaic augmentation
```

### For Faster Training

```yaml
epochs: 50            # Fewer epochs
batch: 32             # Larger batch (if GPU allows)
workers: 8            # More workers
cache: true           # Cache images (if enough RAM)
```

### For Raspberry Pi Optimization

```yaml
imgsz: 256            # Smaller images
model: yolov8n.pt     # Smallest model
batch: 16             # Moderate batch size
```

## Performance Benchmarks

### Expected Training Time

| GPU | Batch Size | Images | Time/Epoch | Total Time |
|-----|------------|--------|------------|------------|
| RTX 3060 | 16 | 500 | ~30s | ~50 min |
| RTX 3080 | 32 | 500 | ~20s | ~33 min |
| CPU | 8 | 500 | ~5min | ~8 hours |

### Expected Results

| Dataset Size | mAP@0.5 | mAP@0.5:0.95 |
|--------------|---------|--------------|
| 300 images | 0.65-0.75 | 0.40-0.50 |
| 500 images | 0.75-0.85 | 0.50-0.60 |
| 1000+ images | 0.85-0.95 | 0.60-0.75 |

## Next Steps

After successful training:

1. **Export model**: Convert to ONNX for deployment
2. **Test integration**: Run YOLO+ResNet pipeline
3. **Benchmark**: Test FPS on target hardware
4. **Deploy**: Transfer to Raspberry Pi

See [deployment_guide.md](deployment_guide.md) for deployment instructions.
