# TILDA Dataset Guide

The TILDA (Textured Images from the Lab for Document Analysis) dataset is a generic texture database.

## 1. Installation
Ensure you have the environment set up:
```bash
conda activate texvision
pip install -r requirements.txt

# For GPU Support (Important!):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
## 2. Download
You have mentioned you already downloaded it. TILDA typically consists of multiple `.zip` files or a large archive containing folders named `c1` through `c8` (representing different textures like silk, paper, etc.) and various environmental conditions (e.g., `e1` to `e4` for illumination variations).

Common structure:
```
tilda/
  c1/
    img1.bmp
    ...
  c2/
  ...
```

## 2. Preparation
TexVision-Pro requires data in a standard `train/val/test` hierarchy. Use the provided script to reorganize your downloaded data.

**Usage:**

```bash
# If you have a zip file
python scripts/prepare_tilda.py --input /path/to/download/tilda.zip --output data/tilda

# If you hava an extracted folder
python scripts/prepare_tilda.py --input /path/to/downloaded_folder --output data/tilda
```

This script will:
1. Extract the zip (if applicable).
2. Randomly split images into Train (70%), Val (15%), and Test (15%).
3. Organize them into `data/tilda/train/<class_name>`, `data/tilda/val/<class_name>`, etc.

## 3. Training
To train on TILDA, first update `configs/datasets.yaml` to point to the correct split ratios if you changed them (defaults are matched).

Then run training with a config override or simply modify `configs/training.yaml` (if you want to make it default):

**Option 1: Command Line Override (requires updating train.py to accept dataset arg, currently hardcoded in yaml)**
Modify `configs/datasets.yaml`:
- Ensure `tilda` block is correct.

**Option 2: Update Training Script or Manual Load**
Currently, `train.py` loads `dtd` by default from the config logic in `get_loss` / dataset section.
**You must ensure `train.py` loads the `tilda` dataset.**

*Note: The current `train.py` has a hardcoded reference to `dtd` in the dataset loading section. You should update `train.py` to key off a config value.*

## 4. Updates Required for `train.py`
To support easy switching, modify `train.py`:
Change:
```python
data_dir = dataset_cfg['datasets']['dtd']['root_dir']
```
To:
```python
dataset_name = training_cfg.get('dataset', 'dtd') # Add this field to your training yaml or arg
data_dir = dataset_cfg['datasets'][dataset_name]['root_dir']
```
