#!/bin/bash

# Define environment name
ENV_NAME="texvision"

echo "Creating Conda environment: $ENV_NAME"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Please install Anaconda or Miniconda."
    exit 1
fi

# Create environment with Python 3.9
conda create -n $ENV_NAME python=3.9 -y

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Installing dependencies..."

# Install PyTorch with CUDA support (adjust cuda version as needed)
# Assuming CUDA 11.8 for broad compatibility, user can adjust
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other requirements
pip install -r ../requirements.txt

echo "Environment setup complete. Activate with: conda activate $ENV_NAME"
