"""
Create sample YOLO annotations to demonstrate the format
This helps verify the annotation pipeline before manual annotation begins
"""

import os
from pathlib import Path
import random

def create_sample_annotation(image_path: Path, output_dir: Path):
    """
    Create a sample YOLO format annotation
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1]
    """
    # Get image name
    image_name = image_path.stem
    
    # Create label file
    label_file = output_dir / f"{image_name}.txt"
    
    # Create sample annotations (1-2 defects per image)
    num_defects = random.randint(1, 2)
    
    annotations = []
    for _ in range(num_defects):
        # Random class (0-3: hole, objects, oil spot, thread error)
        class_id = random.randint(0, 3)
        
        # Random bounding box (normalized coordinates)
        x_center = random.uniform(0.2, 0.8)
        y_center = random.uniform(0.2, 0.8)
        width = random.uniform(0.1, 0.3)
        height = random.uniform(0.1, 0.3)
        
        # Ensure box stays within image bounds
        x_center = max(width/2, min(1 - width/2, x_center))
        y_center = max(height/2, min(1 - height/2, y_center))
        
        annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Write to file
    with open(label_file, 'w') as f:
        f.write('\n'.join(annotations))
    
    return len(annotations)


def main():
    """Create sample annotations for demonstration"""
    
    # Paths
    base_dir = Path('data/TILDA_yolo')
    
    print("Creating sample annotations for demonstration...")
    print("="*60)
    
    total_annotations = 0
    
    for split in ['train', 'val', 'test']:
        images_dir = base_dir / split / 'images'
        labels_dir = base_dir / split / 'labels'
        
        # Get first 5 images from each split (support both jpg and png)
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        image_files = image_files[:5]
        
        print(f"\n{split.upper()} split:")
        print(f"Creating {len(image_files)} sample annotations...")
        
        for img_path in image_files:
            num_boxes = create_sample_annotation(img_path, labels_dir)
            total_annotations += num_boxes
            print(f"  ✓ {img_path.name}: {num_boxes} defects")
    
    print("\n" + "="*60)
    print(f"✅ Created {total_annotations} sample annotations")
    print("="*60)
    print("\nThese are SAMPLE annotations for demonstration only.")
    print("You must replace them with real annotations using LabelImg.")
    print("\nNext steps:")
    print("1. Run: labelImg")
    print("2. Annotate all images in train/val/test")
    print("3. Validate: python scripts/validate_annotations.py")
    print("4. Train: python detection/yolo_trainer.py")


if __name__ == "__main__":
    main()
