# 🚀 Quick Start Guide - YOLOv8 Fabric Defect Detection

Get started with YOLOv8 defect detection in 5 simple steps!

## Prerequisites

- ✅ ResNet-50 trained (96.25% accuracy)
- ✅ TILDA dataset available
- ✅ YOLOv8 dependencies installed

## Step 1: Verify Installation

```bash
# Check if YOLOv8 is installed
python -c "from ultralytics import YOLO; print('✅ YOLOv8 installed')"

# Check if LabelImg is installed
labelImg --version || echo "❌ Install with: pip install labelImg"
```

## Step 2: Prepare Dataset (Already Done! ✅)

Your dataset is ready at `data/TILDA_yolo/`:
- **Train**: 480 images
- **Val**: 160 images  
- **Test**: 160 images
- **Classes**: hole, objects, oil spot, thread error

## Step 3: Annotate Images

### Start LabelImg

```bash
labelImg
```

### Configure LabelImg

1. Click **"PascalVOC"** button → Changes to **"YOLO"** ✅
2. **Open Dir**: `data/TILDA_yolo/train/images`
3. **Change Save Dir**: `data/TILDA_yolo/train/labels`

### Annotate

- Press `W` to create bounding box
- Draw box around defect
- Select class (hole, objects, oil spot, thread error)
- Press `Ctrl+S` to save
- Press `D` for next image

**Target**: 300-500 annotated images

See [docs/annotation_guide.md](docs/annotation_guide.md) for detailed instructions.

## Step 4: Validate Annotations

After annotating 50+ images:

```bash
python scripts/validate_annotations.py \
    --data data/TILDA_yolo/data.yaml \
    --split train \
    --plot
```

**Expected**: No errors, balanced class distribution

## Step 5: Train YOLOv8

Once you have 300+ annotated images:

```bash
python detection/yolo_trainer.py \
    --config detection/configs/yolo_training.yaml \
    --validate \
    --export onnx
```

**Training Time**: 2-4 hours (GPU)

**Monitor Progress**:
```bash
tensorboard --logdir detection/runs/detect/train
```

## Step 6: Test Detection

```bash
python detection/yolo_inference.py \
    --model detection/runs/detect/train/weights/best.onnx \
    --source data/TILDA_yolo/test/images/hole_001.jpg \
    --save outputs/test_result.jpg
```

## Step 7: Test Integrated Pipeline

```bash
python pipeline/integrated_detector.py \
    --yolo-model detection/runs/detect/train/weights/best.onnx \
    --resnet-model checkpoints/best_model.pth \
    --source test_image.jpg \
    --save-dir outputs/integrated
```

## Step 8: Deploy to Raspberry Pi

### On Raspberry Pi:

```bash
# 1. Setup
./deployment/rpi_setup.sh

# 2. Transfer models (from PC)
# scp models from PC to RPi

# 3. Run web interface
python deployment/flask_app.py \
    --yolo-model models/best.onnx \
    --resnet-model models/best_model.pth
```

Access: `http://raspberrypi.local:5000`

---

## 📚 Documentation

| Guide | Purpose |
|-------|---------|
| [annotation_guide.md](docs/annotation_guide.md) | How to annotate with LabelImg |
| [yolo_training_guide.md](docs/yolo_training_guide.md) | Training instructions |
| [deployment_guide.md](docs/deployment_guide.md) | Raspberry Pi deployment |

---

## 🎯 Current Status

- ✅ Phase 1: Setup complete
- ✅ Phase 2: Dataset prepared (800 images)
- ✅ Phase 3: Training scripts ready
- ✅ Phase 4: Export tools ready
- ✅ Phase 5: Integration pipeline ready
- ✅ Phase 6: Deployment tools ready
- 📝 **Next**: Annotate images with LabelImg

---

## ⚡ Quick Commands

```bash
# Annotate
labelImg

# Validate
python scripts/validate_annotations.py --data data/TILDA_yolo/data.yaml

# Train
python detection/yolo_trainer.py --config detection/configs/yolo_training.yaml

# Test
python detection/yolo_inference.py --model best.onnx --source test.jpg

# Deploy
python deployment/flask_app.py --yolo-model best.onnx --resnet-model best_model.pth
```

---

## 🆘 Need Help?

- **Annotation**: See [annotation_guide.md](docs/annotation_guide.md)
- **Training**: See [yolo_training_guide.md](docs/yolo_training_guide.md)
- **Deployment**: See [deployment_guide.md](docs/deployment_guide.md)

**Ready to start annotating! 🎯**
