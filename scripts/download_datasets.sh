#!/bin/bash

# Dataset Download Script for TexVision-Pro

DATA_DIR="data"
mkdir -p $DATA_DIR

# Download DTD (Describable Textures Dataset)
echo "Downloading DTD..."
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz -P $DATA_DIR
tar -xvf $DATA_DIR/dtd-r1.0.1.tar.gz -C $DATA_DIR
rm $DATA_DIR/dtd-r1.0.1.tar.gz

# KTH-TIPS (Example link, usually requires agreement or specific URL)
# echo "Downloading KTH-TIPS..."
# wget http://www.nada.kth.se/cvap/databases/kth-tips/kth_tips_col_200x200.tar -P $DATA_DIR
# tar -xvf $DATA_DIR/kth_tips_col_200x200.tar -C $DATA_DIR/kth_tips
# rm $DATA_DIR/kth_tips_col_200x200.tar

echo "Datasets downloaded to ./data"
