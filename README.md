# TexVision-Pro

**TexVision-Pro** is an end-to-end AI-powered fabric defect detection and classification system. It combines YOLOv8 object detection with ResNet-50 classification to identify, localize, and classify fabric defects in real-time. The system is optimized for deployment on Raspberry Pi 4 for production line integration.

## Features

- **Defect Detection**: YOLOv8-based real-time defect localization
- **Defect Classification**: ResNet-50 for accurate defect type identification
- **Integrated Pipeline**: Combined YOLO+ResNet for comprehensive analysis
- **Explainability**: Grad-CAM saliency maps and uncertainty quantification
- **Raspberry Pi Deployment**: Optimized for edge deployment on RPi4
- **Web Interface**: Flask-based UI for manual testing
- **Real-time Processing**: Camera feed processing with visual alerts
- **Modularity**: Easy switching between models and configurations

## Supported Defect Types

- **Holes**: Tears and holes in fabric
- **Objects**: Foreign objects on fabric surface
- **Oil Spots**: Oil stains and spots
- **Thread Errors**: Thread defects and loose threads

## Project Structure

```
TexVision-Pro/
├── configs/            # YAML Configuration files
├── data/               # Datasets and database
│   ├── TILDA/         # Classification dataset
│   ├── TILDA_yolo/    # Detection dataset (YOLO format)
│   └── texvision.db   # Inspection database
├── logs/               # Run logs, plots, and visualizations
├── models/             # Model weights and checkpoints
│   ├── yolo11n.pt
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   └── checkpoints/   # ResNet model checkpoints
├── src/                # Core source code
│   ├── detection/     # YOLOv8 detection module
│   ├── pipeline/      # Integrated YOLO+ResNet pipeline
│   ├── deployment/    # Flask app and camera processor
│   ├── evaluation/    # Metrics and Explainability
│   └── training/      # Model training scripts
├── scripts/            # CLI Utility scripts
├── docs/               # Documentation
├── LICENSE
├── README.md
└── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/TexVision-Pro.git
cd TexVision-Pro
```

### 2. Setup Environment

```bash
# Create conda environment
bash scripts/setup_env.sh

# Activate environment
conda activate texvision

# IMPORTANT: For GPU support (CUDA 11.8), run:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Datasets

```bash
bash scripts/download_datasets.sh
```

## Quick Start

### Classification (ResNet-50)

**Training:**
```bash
python training/train.py --config configs/training.yaml
```

**Evaluation:**
```bash
python training/validate.py --checkpoint checkpoints/best_model.pth
```

### Detection (YOLOv8)

**Prepare Dataset:**
```bash
python scripts/setup_yolo_dataset.py --tilda-dir data/TILDA
```

**Annotate Images:**
```bash
labelImg
# See docs/annotation_guide.md for detailed instructions
```

**Train YOLOv8:**
```bash
python detection/yolo_trainer.py --config detection/configs/yolo_training.yaml
```

**Test Detection:**
```bash
python detection/yolo_inference.py \
    --model detection/runs/detect/train/weights/best.pt \
    --source test_image.jpg
```

### Integrated Pipeline (YOLO + ResNet)

```bash
python pipeline/integrated_detector.py \
    --yolo-model detection/runs/detect/train/weights/best.onnx \
    --resnet-model checkpoints/best_model.pth \
    --source test_image.jpg
```

## Deployment on Raspberry Pi 4

### 1. Setup Raspberry Pi

```bash
# On Raspberry Pi
./deployment/rpi_setup.sh
```

### 2. Transfer Models

```bash
# On PC
scp detection/runs/detect/train/weights/best.onnx pi@raspberrypi:~/models/
scp checkpoints/best_model.pth pi@raspberrypi:~/models/
```

### 3. Run Web Interface

```bash
# On Raspberry Pi or local
python src/deployment/flask_app.py \
    --yolo-model models/yolov8s.pt \
    --resnet-model models/checkpoints/best_model.pth
```

Access at: `http://raspberrypi.local:5000`

### 4. Run Real-time Detection

```bash
python deployment/camera_processor.py \
    --yolo-model models/best.onnx \
    --resnet-model models/best_model.pth \
    --camera 0
```

See [deployment_guide.md](docs/deployment_guide.md) for detailed instructions.

## Results

### ResNet-50 Classification
- **Accuracy**: 96.25%
- **Precision**: 96.05%
- **Recall**: 96.25%
- **Model**: `checkpoints/best_model.pth`

### YOLOv8n Detection (Expected)
- **mAP@0.5**: 0.75-0.85
- **mAP@0.5:0.95**: 0.50-0.60
- **FPS on RPi4**: 8-15 (YOLO only), 5-10 (YOLO+ResNet)

### Integrated System
- **Overall Accuracy**: >90%
- **Real-time Processing**: ✅
- **Raspberry Pi Compatible**: ✅

## Documentation

- [Annotation Guide](docs/annotation_guide.md) - How to annotate images with LabelImg
- [Training Guide](docs/yolo_training_guide.md) - YOLOv8 training instructions
- [Deployment Guide](docs/deployment_guide.md) - Raspberry Pi deployment
- [TILDA Dataset Guide](docs/tilda_guide.md) - Dataset information

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{texvision-pro,
  title={TexVision-Pro: AI-Powered Fabric Defect Detection},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  url={https://github.com/yourusername/TexVision-Pro}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TILDA Fabric Defect Dataset
- Ultralytics YOLOv8
- PyTorch and torchvision
- Raspberry Pi Foundation
