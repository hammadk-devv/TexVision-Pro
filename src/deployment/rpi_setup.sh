#!/bin/bash
# Raspberry Pi 4 Setup Script for TexVision-Pro Deployment
# Run this script on your Raspberry Pi 4

set -e  # Exit on error

echo "=========================================="
echo "TexVision-Pro Raspberry Pi 4 Setup"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libharfbuzz0b \
    libwebp6 \
    libjasper1 \
    libilmbase23 \
    libopenexr23 \
    libgstreamer1.0-0 \
    libavcodec-extra58 \
    libavformat58 \
    libswscale5 \
    libqtgui4 \
    libqt4-test \
    libportaudio2 \
    libsndfile1

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv ~/texvision_env
source ~/texvision_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install numpy==1.23.5  # Compatible with RPi
pip install opencv-python-headless
pip install flask flask-cors
pip install pyyaml
pip install pillow

# Install ONNX Runtime for ARM
echo "Installing ONNX Runtime for ARM..."
pip install onnxruntime

# Install PyTorch for ARM (CPU only)
echo "Installing PyTorch for ARM..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install PiCamera2 (if using Raspberry Pi Camera)
echo "Installing PiCamera2..."
sudo apt-get install -y python3-picamera2

# Enable camera interface
echo "Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Create deployment directory
echo "Creating deployment directory..."
mkdir -p ~/texvision_deployment
cd ~/texvision_deployment

# Set permissions
echo "Setting permissions..."
sudo usermod -a -G video $USER

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy your trained models to ~/texvision_deployment/models/"
echo "   - YOLO model: best.onnx"
echo "   - ResNet model: best_model.pth"
echo "2. Copy deployment scripts to ~/texvision_deployment/"
echo "3. Activate environment: source ~/texvision_env/bin/activate"
echo "4. Run the application: python camera_processor.py"
echo ""
echo "Note: You may need to reboot for camera changes to take effect"
echo "=========================================="
