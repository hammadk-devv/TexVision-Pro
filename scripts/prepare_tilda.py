import os
import shutil
import zipfile
import argparse
import random
from pathlib import Path

def setup_tilda_structure(data_dir, output_dir, split_ratios=(0.7, 0.15, 0.15)):
    """
    Organizes TILDA dataset from raw ZIP or folder into PyTorch ImageFolder structure.
    TILDA structure often comes as 'c1', 'c2' ... folders with images.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(output_dir / split, exist_ok=True)
    
    # Extract if zip
    if data_dir.suffix == '.zip':
        print(f"Extracting {data_dir}...")
        extract_path = output_dir / "raw_extracted"
        with zipfile.ZipFile(data_dir, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        source_dir = extract_path
    else:
        source_dir = data_dir

    # Inspect structure
    print(f"Inspecting structure in: {source_dir}")
    all_dirs_with_images = []
    
    # Walk through directory
    for root, dirs, files in os.walk(source_dir):
        # Check for images in this directory
        images = [f for f in files if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        if images:
            all_dirs_with_images.append(Path(root))

    if not all_dirs_with_images:
        print("ERROR: No images found in the extracted directory!")
        # Print first few files found to help debug
        print("First 10 files found in directory:")
        count = 0
        for root, dirs, files in os.walk(source_dir):
            for f in files:
                print(os.path.join(root, f))
                count += 1
                if count >= 10: break
            if count >= 10: break
        return

    print(f"Found {len(all_dirs_with_images)} folders containing images.")
    
    # Filter for class folders
    # First try strict 'c' or 'texture' naming
    class_folders = [p for p in all_dirs_with_images if p.name.lower().startswith('c') or 'texture' in p.name.lower()]
    
    # If that fails, assume ALL folders with images are classes (fallback)
    if not class_folders and all_dirs_with_images:
        print("Warning: Strict class naming (c*, texture*) not matched. Using all folders with images as classes.")
        class_folders = all_dirs_with_images

    # Dedup and sort
    class_folders = sorted(list(set(class_folders)), key=lambda p: p.name)
    
    if not class_folders:
        print("No class folders found! Please check the structure of your TILDA download.")
        print("Expected structure: root/c1/*.bmp, root/c2/*.bmp, etc.")
        return

    print(f"Found {len(class_folders)} classes: {[f.name for f in class_folders]}")

    for class_folder in class_folders:
        class_name = class_folder.name
        images = list(class_folder.glob('*.bmp')) + list(class_folder.glob('*.png')) + list(class_folder.glob('*.jpg'))
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        # n_test = rest
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }
        
        for split, split_imgs in splits.items():
            split_class_dir = output_dir / split / class_name
            os.makedirs(split_class_dir, exist_ok=True)
            
            for img_path in split_imgs:
                shutil.copy(img_path, split_class_dir / img_path.name)
                
        print(f"Processed class {class_name}: {n_total} images")

    # Cleanup extraction if used
    if data_dir.suffix == '.zip' and (output_dir / "raw_extracted").exists():
        shutil.rmtree(output_dir / "raw_extracted")

    print(f"Dataset prepared at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TILDA dataset for PyTorch ImageFolder")
    parser.add_argument("--input", type=str, required=True, help="Path to TILDA zip file or root directory containing class folders")
    parser.add_argument("--output", type=str, default="data/tilda", help="Output directory for processed data")
    
    args = parser.parse_args()
    
    setup_tilda_structure(args.input, args.output)
