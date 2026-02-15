"""
Validate YOLO annotations
Checks format correctness, generates statistics, and identifies issues
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np


class AnnotationValidator:
    """Validate YOLO format annotations"""
    
    def __init__(self, data_yaml_path: str):
        """
        Initialize validator
        
        Args:
            data_yaml_path: Path to data.yaml configuration file
        """
        with open(data_yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config['names']
        self.num_classes = self.config['nc']
        
        self.stats = {
            'train': defaultdict(int),
            'val': defaultdict(int),
            'test': defaultdict(int)
        }
        
        self.errors = []
        
    def validate_split(self, split: str = 'train') -> Dict:
        """
        Validate annotations for a specific split
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            
        Returns:
            Dictionary with validation results
        """
        print(f"\n{'='*60}")
        print(f"Validating {split.upper()} split")
        print(f"{'='*60}")
        
        # Get splits from config
        splits_cfg = {
            'train': self.config.get('train'),
            'val': self.config.get('val'),
            'test': self.config.get('test')
        }
        
        split_path = splits_cfg.get(split)
        if not split_path:
            print(f"❌ Split '{split}' not found in config")
            return {}
            
        # Handle if path is a .txt file listing images or a directory
        split_path = Path(split_path)
        image_files = []
        
        if split_path.suffix == '.txt':
            # It's a list of images
            with open(split_path, 'r') as f:
                image_files = [Path(line.strip()) for line in f if line.strip()]
            labels_dir = split_path.parent / 'labels'
        else:
            # Assume it's a directory structure as per ultralytics
            base_path = Path(self.config.get('path', ''))
            images_dir = base_path / split / 'images'
            labels_dir = base_path / split / 'labels'
            
            if not images_dir.exists():
                images_dir = split_path / 'images'
                labels_dir = split_path / 'labels'
                
            if not images_dir.exists():
                print(f"❌ Images directory not found for split {split}")
                return {}
            
            image_files = list(images_dir.glob('*.jpg')) + \
                         list(images_dir.glob('*.png')) + \
                         list(images_dir.glob('*.jpeg'))
        
        print(f"Found {len(image_files)} images")
        
        # Statistics
        total_images = len(image_files)
        annotated_images = 0
        total_boxes = 0
        class_counts = defaultdict(int)
        box_sizes = []
        images_without_labels = []
        invalid_annotations = []
        
        # Validate each image
        for img_path in image_files:
            # Get corresponding label file
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                images_without_labels.append(img_path.name)
                continue
            
            annotated_images += 1
            
            # Read and validate annotations
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    
                    # Validate format
                    if len(parts) != 5:
                        invalid_annotations.append({
                            'file': label_path.name,
                            'line': line_num,
                            'error': f'Invalid format: expected 5 values, got {len(parts)}'
                        })
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                    except ValueError as e:
                        invalid_annotations.append({
                            'file': label_path.name,
                            'line': line_num,
                            'error': f'Invalid values: {e}'
                        })
                        continue
                    
                    # Validate class ID
                    if class_id < 0 or class_id >= self.num_classes:
                        invalid_annotations.append({
                            'file': label_path.name,
                            'line': line_num,
                            'error': f'Invalid class ID: {class_id} (must be 0-{self.num_classes-1})'
                        })
                        continue
                    
                    # Validate coordinates (should be normalized 0-1)
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                           0 < width <= 1 and 0 < height <= 1):
                        invalid_annotations.append({
                            'file': label_path.name,
                            'line': line_num,
                            'error': f'Coordinates out of range: ({x_center}, {y_center}, {width}, {height})'
                        })
                        continue
                    
                    # Update statistics
                    total_boxes += 1
                    class_counts[class_id] += 1
                    box_sizes.append((width, height))
                    
            except Exception as e:
                invalid_annotations.append({
                    'file': label_path.name,
                    'line': 0,
                    'error': f'Error reading file: {e}'
                })
        
        # Print results
        print(f"\n📊 Validation Results:")
        print(f"  Total images: {total_images}")
        print(f"  Annotated images: {annotated_images} ({annotated_images/total_images*100:.1f}%)")
        print(f"  Images without labels: {len(images_without_labels)}")
        print(f"  Total bounding boxes: {total_boxes}")
        print(f"  Avg boxes per image: {total_boxes/annotated_images if annotated_images > 0 else 0:.2f}")
        
        # Class distribution
        print(f"\n📈 Class Distribution:")
        for class_id in range(self.num_classes):
            count = class_counts[class_id]
            percentage = count / total_boxes * 100 if total_boxes > 0 else 0
            class_name = self.class_names[class_id]
            print(f"  {class_name:15s} (ID {class_id}): {count:4d} boxes ({percentage:5.1f}%)")
        
        # Box size statistics
        if box_sizes:
            widths = [w for w, h in box_sizes]
            heights = [h for w, h in box_sizes]
            
            print(f"\n📏 Bounding Box Size Statistics:")
            print(f"  Width  - Min: {min(widths):.3f}, Max: {max(widths):.3f}, Avg: {np.mean(widths):.3f}")
            print(f"  Height - Min: {min(heights):.3f}, Max: {max(heights):.3f}, Avg: {np.mean(heights):.3f}")
        
        # Errors
        if images_without_labels:
            print(f"\n⚠️  Images without labels ({len(images_without_labels)}):")
            for img_name in images_without_labels[:10]:  # Show first 10
                print(f"  - {img_name}")
            if len(images_without_labels) > 10:
                print(f"  ... and {len(images_without_labels) - 10} more")
        
        if invalid_annotations:
            print(f"\n❌ Invalid annotations ({len(invalid_annotations)}):")
            for error in invalid_annotations[:10]:  # Show first 10
                print(f"  - {error['file']} (line {error['line']}): {error['error']}")
            if len(invalid_annotations) > 10:
                print(f"  ... and {len(invalid_annotations) - 10} more")
        
        # Overall status
        print(f"\n{'='*60}")
        if len(invalid_annotations) == 0 and len(images_without_labels) == 0:
            print("✅ All annotations are valid!")
        elif len(invalid_annotations) == 0:
            print("⚠️  Some images are missing annotations")
        else:
            print("❌ Found annotation errors - please fix them")
        print(f"{'='*60}")
        
        # Store statistics
        self.stats[split] = {
            'total_images': total_images,
            'annotated_images': annotated_images,
            'total_boxes': total_boxes,
            'class_counts': dict(class_counts),
            'box_sizes': box_sizes,
            'images_without_labels': images_without_labels,
            'invalid_annotations': invalid_annotations
        }
        
        return self.stats[split]
    
    def validate_all(self):
        """Validate all splits"""
        for split in ['train', 'val', 'test']:
            self.validate_split(split)
    
    def plot_statistics(self, save_dir: str = 'logs/annotation_stats'):
        """
        Generate visualization plots for annotation statistics
        
        Args:
            save_dir: Directory to save plots
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Class distribution across splits
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, split in enumerate(['train', 'val', 'test']):
            if split not in self.stats or not self.stats[split]:
                continue
            
            class_counts = self.stats[split].get('class_counts', {})
            if not class_counts:
                continue
            
            classes = [self.class_names[i] for i in range(self.num_classes)]
            counts = [class_counts.get(i, 0) for i in range(self.num_classes)]
            
            axes[idx].bar(classes, counts, color='skyblue')
            axes[idx].set_title(f'{split.capitalize()} Split')
            axes[idx].set_xlabel('Class')
            axes[idx].set_ylabel('Count')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path / 'class_distribution.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved class distribution plot: {save_path / 'class_distribution.png'}")
        plt.close()
        
        # Plot 2: Box size distribution
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, split in enumerate(['train', 'val', 'test']):
            if split not in self.stats or not self.stats[split]:
                continue
            
            box_sizes = self.stats[split].get('box_sizes', [])
            if not box_sizes:
                continue
            
            widths = [w for w, h in box_sizes]
            heights = [h for w, h in box_sizes]
            
            axes[idx].scatter(widths, heights, alpha=0.5, s=10)
            axes[idx].set_title(f'{split.capitalize()} Split')
            axes[idx].set_xlabel('Width (normalized)')
            axes[idx].set_ylabel('Height (normalized)')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'box_sizes.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved box size plot: {save_path / 'box_sizes.png'}")
        plt.close()
        
        print(f"\n✓ All plots saved to: {save_path}")


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate YOLO annotations')
    parser.add_argument('--data', type=str,
                       default='data/TILDA_yolo/data.yaml',
                       help='Path to data.yaml file')
    parser.add_argument('--split', type=str,
                       choices=['train', 'val', 'test', 'all'],
                       default='all',
                       help='Which split to validate')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Check if data.yaml exists
    if not os.path.exists(args.data):
        print(f"❌ Error: data.yaml not found at {args.data}")
        print("Please run setup_yolo_dataset.py first")
        return
    
    # Initialize validator
    validator = AnnotationValidator(args.data)
    
    # Validate
    if args.split == 'all':
        validator.validate_all()
    else:
        validator.validate_split(args.split)
    
    # Generate plots
    if args.plot:
        validator.plot_statistics()


if __name__ == "__main__":
    main()
