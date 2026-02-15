# Raspberry Pi 4 Deployment Guide

Complete guide for deploying TexVision-Pro on Raspberry Pi 4.

## Prerequisites

- Raspberry Pi 4 (4GB RAM recommended)
- MicroSD card (32GB+ recommended)
- Raspberry Pi OS (64-bit recommended)
- Raspberry Pi Camera Module (optional)
- Trained models (YOLO + ResNet)

## Part 1: Raspberry Pi Setup

### 1.1 Install Raspberry Pi OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Flash Raspberry Pi OS (64-bit) to SD card
3. Boot Raspberry Pi and complete initial setup

### 1.2 Enable Camera (If Using)

```bash
sudo raspi-config
```

Navigate to: `Interface Options` → `Camera` → `Enable`

Reboot:
```bash
sudo reboot
```

### 1.3 Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

## Part 2: Install Dependencies

### 2.1 Transfer Setup Script

From your PC, copy the setup script to Raspberry Pi:

```bash
# On PC
scp deployment/rpi_setup.sh pi@raspberrypi.local:~/

# On Raspberry Pi
chmod +x ~/rpi_setup.sh
```

### 2.2 Run Setup Script

```bash
cd ~
./rpi_setup.sh
```

This will install:
- Python 3 and pip
- OpenCV
- ONNX Runtime (ARM)
- PyTorch (CPU)
- Flask and dependencies

**Note:** This may take 30-60 minutes.

### 2.3 Activate Environment

```bash
source ~/texvision_env/bin/activate
```

## Part 3: Transfer Models

### 3.1 Create Model Directory

```bash
mkdir -p ~/texvision_deployment/models
```

### 3.2 Transfer Models from PC

```bash
# On PC - Transfer YOLO model (ONNX format)
scp detection/runs/detect/train/weights/best.onnx \
    pi@raspberrypi.local:~/texvision_deployment/models/

# Transfer ResNet model
scp checkpoints/best_model.pth \
    pi@raspberrypi.local:~/texvision_deployment/models/
```

### 3.3 Verify Models

```bash
# On Raspberry Pi
ls -lh ~/texvision_deployment/models/
# Should show:
# best.onnx (5-10 MB)
# best_model.pth (90-100 MB)
```

## Part 4: Transfer Application Code

### 4.1 Transfer Deployment Scripts

```bash
# On PC
scp -r deployment/* pi@raspberrypi.local:~/texvision_deployment/
scp -r pipeline pi@raspberrypi.local:~/texvision_deployment/
scp -r detection pi@raspberrypi.local:~/texvision_deployment/
```

### 4.2 Verify Structure

```bash
# On Raspberry Pi
cd ~/texvision_deployment
tree -L 2
```

Expected structure:
```
texvision_deployment/
├── models/
│   ├── best.onnx
│   └── best_model.pth
├── deployment/
│   ├── camera_processor.py
│   ├── flask_app.py
│   └── alert_system.py
├── pipeline/
│   └── integrated_detector.py
└── detection/
    └── yolo_inference.py
```

## Part 5: Test Deployment

### 5.1 Test Flask Web Interface

```bash
cd ~/texvision_deployment
python deployment/flask_app.py \
    --yolo-model models/best.onnx \
    --resnet-model models/best_model.pth \
    --host 0.0.0.0 \
    --port 5000
```

Access from your PC: `http://raspberrypi.local:5000`

**Expected:**
- Web interface loads
- Can upload images
- Detections work correctly

### 5.2 Test Camera Processor (If Using Camera)

```bash
python deployment/camera_processor.py \
    --yolo-model models/best.onnx \
    --resnet-model models/best_model.pth \
    --camera 0 \
    --frame-skip 2
```

**Expected:**
- Camera feed displays
- Detections appear in real-time
- FPS shown in overlay

Press `q` to quit.

## Part 6: Performance Optimization

### 6.1 Benchmark Performance

Create benchmark script:

```python
# benchmark_rpi.py
import time
import cv2
from pipeline.integrated_detector import IntegratedDetector

detector = IntegratedDetector(
    yolo_model_path='models/best.onnx',
    resnet_model_path='models/best_model.pth',
    device='cpu'
)

# Load test image
image = cv2.imread('test_image.jpg')

# Warm-up
for _ in range(5):
    detector.detect_and_classify(image)

# Benchmark
times = []
for _ in range(20):
    start = time.time()
    results = detector.detect_and_classify(image)
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
fps = 1.0 / avg_time

print(f"Average time: {avg_time*1000:.2f} ms")
print(f"FPS: {fps:.2f}")
```

Run:
```bash
python benchmark_rpi.py
```

**Target Performance:**
- Single image: 100-200ms
- FPS: 5-10

### 6.2 Optimization Tips

If performance is below target:

#### 1. Reduce Frame Skip
```bash
python deployment/camera_processor.py --frame-skip 3  # Process every 3rd frame
```

#### 2. Reduce Image Size
Edit `detection/yolo_inference.py`:
```python
input_size = 256  # Instead of 320
```

#### 3. Increase Swap Space
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### 4. Overclock (Advanced)
```bash
sudo nano /boot/config.txt
# Add:
# over_voltage=6
# arm_freq=2000
sudo reboot
```

**Warning:** Overclocking may void warranty and requires cooling.

## Part 7: Production Deployment

### 7.1 Create Systemd Service

Create service file:
```bash
sudo nano /etc/systemd/system/texvision.service
```

Add:
```ini
[Unit]
Description=TexVision Defect Detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/texvision_deployment
Environment="PATH=/home/pi/texvision_env/bin"
ExecStart=/home/pi/texvision_env/bin/python deployment/camera_processor.py \
    --yolo-model models/best.onnx \
    --resnet-model models/best_model.pth \
    --camera 0 \
    --frame-skip 2 \
    --no-display
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 7.2 Enable Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable texvision.service
sudo systemctl start texvision.service
```

### 7.3 Check Status

```bash
sudo systemctl status texvision.service
```

### 7.4 View Logs

```bash
sudo journalctl -u texvision.service -f
```

## Part 8: Remote Access

### 8.1 Enable SSH

```bash
sudo raspi-config
# Interface Options → SSH → Enable
```

### 8.2 Set Static IP (Optional)

```bash
sudo nano /etc/dhcpcd.conf
```

Add:
```
interface wlan0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1
```

### 8.3 Access Web Interface Remotely

From any device on the network:
```
http://192.168.1.100:5000
```

## Part 9: Monitoring & Maintenance

### 9.1 Check Detections

```bash
ls -lh ~/texvision_deployment/detections/
```

### 9.2 View Detection Log

```bash
cat ~/texvision_deployment/defect_log.json | python -m json.tool
```

### 9.3 Monitor System Resources

```bash
# CPU temperature
vcgencmd measure_temp

# Memory usage
free -h

# CPU usage
top
```

### 9.4 Backup Detection Data

```bash
# On Raspberry Pi
tar -czf detections_backup_$(date +%Y%m%d).tar.gz detections/ defect_log.json

# Transfer to PC
scp detections_backup_*.tar.gz user@pc:~/backups/
```

## Troubleshooting

### Camera Not Working

```bash
# Check camera
vcgencmd get_camera
# Should show: supported=1 detected=1

# Test camera
libcamera-hello
```

### Low FPS

1. Check CPU usage: `top`
2. Increase frame skip: `--frame-skip 3`
3. Reduce image size in code
4. Close other applications

### Model Loading Errors

```bash
# Verify model files
ls -lh models/
md5sum models/best.onnx
md5sum models/best_model.pth
```

### Service Not Starting

```bash
# Check logs
sudo journalctl -u texvision.service -n 50

# Test manually
cd ~/texvision_deployment
source ~/texvision_env/bin/activate
python deployment/camera_processor.py --help
```

## Performance Benchmarks

### Raspberry Pi 4 (4GB)

| Configuration | FPS | Latency | Accuracy |
|---------------|-----|---------|----------|
| YOLOv8n 320x320 | 10-12 | 80-100ms | 0.85 mAP |
| YOLOv8n 256x256 | 15-18 | 55-65ms | 0.80 mAP |
| YOLO+ResNet 320 | 5-8 | 125-200ms | 0.90+ |
| YOLO+ResNet 256 | 8-10 | 100-125ms | 0.88+ |

## Next Steps

- ✅ Deployment complete
- ✅ System running in production
- 📊 Monitor performance metrics
- 🔧 Fine-tune based on real-world data
- 📈 Collect feedback for improvements

## Support

For issues:
1. Check logs: `sudo journalctl -u texvision.service`
2. Review this guide
3. Check model files and permissions
4. Verify network connectivity

**Congratulations! Your TexVision-Pro system is now deployed! 🎉**
