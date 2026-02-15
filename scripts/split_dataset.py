import os
import random

dataset_path = "data/TILDA_yolo"
images_path = os.path.join(dataset_path, "images")
all_images = [f for f in os.listdir(images_path) if f.endswith('.jpg') or f.endswith('.png')]
random.shuffle(all_images)

split_idx = int(len(all_images) * 0.8)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

with open(os.path.join(dataset_path, "train.txt"), "w") as f:
    for img in train_images:
        f.write(os.path.abspath(os.path.join(images_path, img)) + "\n")

with open(os.path.join(dataset_path, "val.txt"), "w") as f:
    for img in val_images:
        f.write(os.path.abspath(os.path.join(images_path, img)) + "\n")

print(f"Dataset split complete: {len(train_images)} train, {len(val_images)} val.")
