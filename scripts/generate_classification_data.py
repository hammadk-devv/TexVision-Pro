import os
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm

def generate_crops(yolo_dir, output_dir, padding=20):
    """
    Generates classification crops from YOLO dataset regions.
    """
    yolo_dir = Path(yolo_dir)
    output_dir = Path(output_dir)
    
    # Load data.yaml
    with open(yolo_dir / 'data.yaml', 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    classes = data_cfg['names']
    print(f"Classes: {classes}")

    # Process Splits
    # Roboflow usually has train, valid, test. Our system expects train, val, test.
    splits_map = {
        'train': 'train',
        'valid': 'val',
        'test': 'test'
    }

    for yolo_split, resnet_split in splits_map.items():
        img_dir = yolo_dir / yolo_split / 'images'
        lbl_dir = yolo_dir / yolo_split / 'labels'
        
        if not img_dir.exists():
            print(f"Skipping missing split: {yolo_split}")
            continue

        print(f"Processing split: {yolo_split} -> {resnet_split}")
        
        images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
        
        for img_path in tqdm(images):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
                
            img = cv2.imread(str(img_path))
            if img is None: continue
            h, w = img.shape[:2]
            
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5: continue
                
                cls_id = int(parts[0])
                cls_name = classes[cls_id]
                
                # YOLO format: cls cx cy nw nh (normalized)
                cx, cy, nw, nh = map(float, parts[1:])
                
                # Convert to pixel coordinates
                x1 = int((cx - nw/2) * w)
                y1 = int((cy - nh/2) * h)
                x2 = int((cx + nw/2) * w)
                y2 = int((cy + nh/2) * h)
                
                # Add padding
                x1_p = max(0, x1 - padding)
                y1_p = max(0, y1 - padding)
                x2_p = min(w, x2 + padding)
                y2_p = min(h, y2 + padding)
                
                crop = img[y1_p:y2_p, x1_p:x2_p]
                
                # Save crop
                save_dir = output_dir / resnet_split / cls_name
                save_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = save_dir / f"{img_path.stem}_crop{i}.jpg"
                cv2.imwrite(str(save_path), crop)

    print(f"✓ Classification dataset generated at {output_dir}")

if __name__ == "__main__":
    generate_crops(
        yolo_dir='data/TILDA_yolo',
        output_dir='data/TILDA'
    )
