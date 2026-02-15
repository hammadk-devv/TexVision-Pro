# Fabric Defect Annotation Guide

This guide provides step-by-step instructions for annotating fabric defect images using LabelImg for YOLOv8 training.

## Table of Contents
1. [Installation](#installation)
2. [Getting Started](#getting-started)
3. [Annotation Guidelines](#annotation-guidelines)
4. [Quality Control](#quality-control)
5. [Tips & Best Practices](#tips--best-practices)

---

## Installation

### Install LabelImg

```bash
pip install labelImg
```

### Verify Installation

```bash
labelImg
```

This should open the LabelImg application.

---

## Getting Started

### 1. Launch LabelImg

```bash
labelImg
```

### 2. Configure LabelImg for YOLO Format

**CRITICAL**: Make sure to set the annotation format to **YOLO**, not PascalVOC!

1. Click **"PascalVOC"** button on the left sidebar
2. It will change to **"YOLO"** - this is correct!
3. The button should show "YOLO" when active

### 3. Set Up Directories

#### For Training Set:
1. Click **"Open Dir"** → Select `data/TILDA_yolo/train/images`
2. Click **"Change Save Dir"** → Select `data/TILDA_yolo/train/labels`

#### For Validation Set (after training set is done):
1. Click **"Open Dir"** → Select `data/TILDA_yolo/val/images`
2. Click **"Change Save Dir"** → Select `data/TILDA_yolo/val/labels`

#### For Test Set (after validation set is done):
1. Click **"Open Dir"** → Select `data/TILDA_yolo/test/images`
2. Click **"Change Save Dir"** → Select `data/TILDA_yolo/test/labels`

### 4. Load Class Names

LabelImg will automatically create a `classes.txt` file in the labels directory. Make sure it contains:

```
hole
objects
oil spot
thread error
```

**Order matters!** The class index in YOLO format corresponds to the line number (0-indexed).

---

## Annotation Guidelines

### Defect Classes

Our dataset has **4 defect classes**:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | hole | Holes or tears in fabric |
| 1 | objects | Foreign objects on fabric |
| 2 | oil spot | Oil stains or spots |
| 3 | thread error | Thread defects, loose threads |

### How to Annotate

#### Step-by-Step Process:

1. **Open Image**: LabelImg loads the first image automatically
2. **Create Bounding Box**: 
   - Press `W` key or click "Create RectBox"
   - Click and drag to draw a box around the defect
   - Make sure the box **tightly** covers the defect
3. **Select Class**: Choose the correct defect class from the popup
4. **Save**: Press `Ctrl+S` or click "Save"
5. **Next Image**: Press `D` key or click "Next Image"

#### Keyboard Shortcuts:
- `W` - Create bounding box
- `D` - Next image
- `A` - Previous image
- `Ctrl+S` - Save
- `Del` - Delete selected box
- `Ctrl+D` - Duplicate box

### Bounding Box Guidelines

#### ✅ Good Annotations:

1. **Tight Fit**: Box should closely fit the defect
   - Include the entire defect
   - Minimize extra background

2. **Single Defect**: One box per defect
   - If multiple defects, create multiple boxes

3. **Clear Boundaries**: Box edges should be clear
   - Align with defect edges
   - No ambiguous boundaries

#### ❌ Bad Annotations:

1. **Too Loose**: Box includes too much background
2. **Too Tight**: Box cuts off part of the defect
3. **Multiple Defects in One Box**: Should be separate boxes
4. **Overlapping Boxes**: Avoid unless defects actually overlap

### Class-Specific Guidelines

#### Holes (Class 0)
- Include the entire hole area
- Include frayed edges if present
- For small holes: ensure box is at least 10x10 pixels

#### Objects (Class 1)
- Include the entire foreign object
- Include shadows if they're part of the defect
- For multiple objects close together: use separate boxes

#### Oil Spots (Class 2)
- Include the entire stained area
- Include faded edges of the stain
- For irregular shapes: box should cover the full extent

#### Thread Errors (Class 3)
- Include the full length of loose/broken threads
- For thread bunching: include the bunched area
- For missing threads: box the affected area

---

## Quality Control

### Self-Check Checklist

Before moving to the next image, verify:

- [ ] All defects in the image are annotated
- [ ] Bounding boxes are tight and accurate
- [ ] Correct class is assigned to each box
- [ ] No overlapping boxes (unless defects overlap)
- [ ] Annotation is saved (check for `.txt` file in labels folder)

### Validation Process

Every 50 images, review your annotations:

1. Open 5 random annotated images
2. Check if boxes are accurate
3. Verify class labels are correct
4. Adjust if needed

### Common Mistakes to Avoid

1. **Wrong Format**: Using PascalVOC instead of YOLO
2. **Wrong Save Directory**: Saving to images folder instead of labels
3. **Missing Defects**: Not annotating all defects in an image
4. **Wrong Class**: Confusing similar defect types
5. **Too Large Boxes**: Including too much background

---

## Tips & Best Practices

### Speed Up Annotation

1. **Use Keyboard Shortcuts**: Learn `W`, `D`, `A`, `Ctrl+S`
2. **Batch Similar Images**: Annotate similar defects together
3. **Take Breaks**: Annotation fatigue leads to errors
4. **Set Goals**: Aim for 50-100 images per session

### Handling Edge Cases

#### Unclear Defects
- If you can't identify the defect type: **skip the image**
- Mark unclear images in a separate list for review

#### Multiple Defects
- Annotate all defects in the image
- Use separate boxes for each defect
- Assign correct class to each box

#### Very Small Defects
- Minimum box size: 10x10 pixels
- If defect is smaller: slightly enlarge the box
- Ensure defect is visible in the box

#### Defects at Image Edges
- Include partial defects at edges
- Box should extend to image boundary
- Don't create boxes outside image bounds

### Progress Tracking

Keep track of your annotation progress:

```
Total Images: 800
- Train: 480 images
- Val: 160 images
- Test: 160 images

Daily Goal: 100 images
Estimated Time: 8 days (at 2 hours/day)
```

Create a simple log:
```
Day 1: 0-100 (train)
Day 2: 100-200 (train)
Day 3: 200-300 (train)
...
```

---

## Annotation Workflow Summary

### Daily Workflow:

1. **Start LabelImg**
   ```bash
   labelImg
   ```

2. **Configure**
   - Set format to YOLO
   - Open images directory
   - Set save directory

3. **Annotate**
   - Draw boxes (`W` key)
   - Select class
   - Save (`Ctrl+S`)
   - Next image (`D` key)

4. **Quality Check**
   - Every 50 images, review 5 random annotations
   - Fix any errors

5. **Track Progress**
   - Update your progress log
   - Note any problematic images

### After Completing a Split:

1. **Validate Annotations**
   ```bash
   python scripts/validate_annotations.py --split train
   ```

2. **Review Statistics**
   - Check class distribution
   - Verify all images have annotations
   - Look for outliers (very large/small boxes)

3. **Move to Next Split**
   - Complete train → val → test in order

---

## Expected Results

### Annotation Statistics (Target)

After annotating all 800 images:

| Split | Images | Avg Defects/Image | Total Boxes |
|-------|--------|-------------------|-------------|
| Train | 480 | 1-3 | ~960-1440 |
| Val | 160 | 1-3 | ~320-480 |
| Test | 160 | 1-3 | ~320-480 |

### Class Distribution (Target)

Aim for balanced distribution:

| Class | Percentage |
|-------|------------|
| hole | ~25% |
| objects | ~25% |
| oil spot | ~25% |
| thread error | ~25% |

---

## Troubleshooting

### LabelImg Not Opening
```bash
# Reinstall
pip uninstall labelImg
pip install labelImg
```

### Wrong Format Saved
- Delete `.xml` files if any
- Ensure "YOLO" button is active
- Re-annotate affected images

### Missing classes.txt
Create manually in labels directory:
```
hole
objects
oil spot
thread error
```

### Annotations Not Saving
- Check write permissions on labels directory
- Ensure save directory is set correctly
- Try saving manually (`Ctrl+S`)

---

## Next Steps

After completing annotations:

1. **Validate Dataset**
   ```bash
   python scripts/validate_annotations.py
   ```

2. **Start Training**
   ```bash
   python detection/yolo_trainer.py --config detection/configs/yolo_training.yaml
   ```

3. **Monitor Progress**
   ```bash
   tensorboard --logdir detection/runs/detect/train
   ```

---

## Questions?

If you encounter issues:
1. Check this guide first
2. Review the [YOLOv8 documentation](https://docs.ultralytics.com/)
3. Verify your annotation format with validation script

**Good luck with annotation! 🎯**
