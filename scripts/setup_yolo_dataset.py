"""
Script to set up YOLO dataset directory structure
Creates the required folders and prepares the dataset for annotation
"""

import os
import shutil
from pathlib import Path
import yaml
import random
from typing import List


def create_yolo_structure(base_dir: str, dataset_name: str = "TILDA_yolo"):
    """
    Create YOLO dataset directory structure
    
    Structure:
    TILDA_yolo/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml
    """
    yolo_dir = Path(base_dir) / dataset_name
    
    # Create directories
    splits = ['train', 'val', 'test']
    subdirs = ['images', 'labels']
    
    print(f"Creating YOLO dataset structure at: {yolo_dir}")
    
    for split in splits:
        for subdir in subdirs:
            dir_path = yolo_dir / split / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {dir_path}")
    
    return yolo_dir


def copy_images_from_tilda(
    tilda_dir: str,
    yolo_dir: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    max_images_per_class: int = None
):
    """
    Copy images from TILDA dataset to YOLO structure
    
    Args:
        tilda_dir: Path to TILDA dataset (with train/val/test subdirs)
        yolo_dir: Path to YOLO dataset directory
        train_ratio: Ratio of images for training
        val_ratio: Ratio of images for validation
        test_ratio: Ratio of images for testing
        max_images_per_class: Maximum images per class (None = all)
    """
    tilda_path = Path(tilda_dir)
    yolo_path = Path(yolo_dir)
    
    # Get all defect classes (exclude 'good' class for annotation)
    defect_classes = []
    for split in ['train', 'val', 'test']:
        split_dir = tilda_path / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir() and class_dir.name != 'good':
                    if class_dir.name not in defect_classes:
                        defect_classes.append(class_dir.name)
    
    print(f"\nFound defect classes: {defect_classes}")
    
    # Collect all defect images
    all_images = []
    for split in ['train', 'val', 'test']:
        split_dir = tilda_path / split
        if not split_dir.exists():
            continue
        
        for class_name in defect_classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            
            # Get image files
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.png')) + \
                         list(class_dir.glob('*.jpeg'))
            
            for img_path in image_files:
                all_images.append({
                    'path': img_path,
                    'class': class_name
                })
    
    print(f"Total defect images found: {len(all_images)}")
    
    # Limit images per class if specified
    if max_images_per_class:
        class_counts = {}
        filtered_images = []
        
        # Shuffle to get random selection
        random.shuffle(all_images)
        
        for img_info in all_images:
            class_name = img_info['class']
            count = class_counts.get(class_name, 0)
            
            if count < max_images_per_class:
                filtered_images.append(img_info)
                class_counts[class_name] = count + 1
        
        all_images = filtered_images
        print(f"Limited to {len(all_images)} images ({max_images_per_class} per class)")
    
    # Shuffle and split
    random.shuffle(all_images)
    
    n_total = len(all_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]
    
    print(f"\nSplit distribution:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    
    # Copy images to YOLO structure
    splits_data = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split_name, images in splits_data.items():
        dest_dir = yolo_path / split_name / 'images'
        
        print(f"\nCopying {len(images)} images to {split_name}...")
        for i, img_info in enumerate(images):
            src_path = img_info['path']
            # Create unique filename with class prefix
            dest_filename = f"{img_info['class']}_{src_path.name}"
            dest_path = dest_dir / dest_filename
            
            shutil.copy2(src_path, dest_path)
            
            if (i + 1) % 50 == 0:
                print(f"  Copied {i + 1}/{len(images)} images...")
        
        print(f"  ✓ Completed {split_name} split")
    
    return defect_classes


def create_data_yaml(yolo_dir: str, class_names: List[str]):
    """
    Create data.yaml configuration file for YOLO training
    
    Args:
        yolo_dir: Path to YOLO dataset directory
        class_names: List of class names
    """
    yolo_path = Path(yolo_dir)
    
    # Create absolute paths
    train_path = (yolo_path / 'train' / 'images').absolute()
    val_path = (yolo_path / 'val' / 'images').absolute()
    test_path = (yolo_path / 'test' / 'images').absolute()
    
    data_config = {
        'path': str(yolo_path.absolute()),
        'train': str(train_path),
        'val': str(val_path),
        'test': str(test_path),
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = yolo_path / 'data.yaml'
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Created data.yaml at: {yaml_path}")
    print(f"\nDataset configuration:")
    print(f"  Classes ({len(class_names)}): {class_names}")
    print(f"  Train path: {train_path}")
    print(f"  Val path: {val_path}")
    print(f"  Test path: {test_path}")
    
    return yaml_path


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup YOLO dataset structure')
    parser.add_argument('--tilda-dir', type=str,
                       default='data/TILDA',
                       help='Path to TILDA dataset')
    parser.add_argument('--output-dir', type=str,
                       default='data',
                       help='Output directory for YOLO dataset')
    parser.add_argument('--dataset-name', type=str,
                       default='TILDA_yolo',
                       help='Name of YOLO dataset folder')
    parser.add_argument('--max-images', type=int,
                       default=None,
                       help='Maximum images per class (for testing)')
    parser.add_argument('--train-ratio', type=float,
                       default=0.6,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float,
                       default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float,
                       default=0.2,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int,
                       default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("="*60)
    print("YOLO Dataset Setup")
    print("="*60)
    
    # Create YOLO directory structure
    yolo_dir = create_yolo_structure(args.output_dir, args.dataset_name)
    
    # Copy images from TILDA
    defect_classes = copy_images_from_tilda(
        tilda_dir=args.tilda_dir,
        yolo_dir=yolo_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_images_per_class=args.max_images
    )
    
    # Create data.yaml
    create_data_yaml(yolo_dir, defect_classes)
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Install LabelImg: pip install labelImg")
    print(f"2. Run LabelImg: labelImg")
    print(f"3. Open directory: {yolo_dir / 'train' / 'images'}")
    print(f"4. Change save dir to: {yolo_dir / 'train' / 'labels'}")
    print(f"5. Set format to 'YOLO' (not PascalVOC)")
    print(f"6. Start annotating!")
    print(f"\nSee docs/annotation_guide.md for detailed instructions")
    print("="*60)


if __name__ == "__main__":
    main()
