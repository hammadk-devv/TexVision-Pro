# 📋 Annotation Workflow - Step-by-Step Instructions

## ✅ LabelImg Should Now Be Open

### Initial Setup (Do Once)

1. **Set Format to YOLO**:
   - Look for button that says "PascalVOC" on the left sidebar
   - Click it - it should change to "YOLO" ✅
   - This is CRITICAL - wrong format will break training

2. **Open Image Directory**:
   - Click "Open Dir" button
   - Navigate to: `D:\Clone FYP\TexVision-Pro\data\TILDA_yolo\train\images`
   - Click "Select Folder"

3. **Set Save Directory**:
   - Click "Change Save Dir" button
   - Navigate to: `D:\Clone FYP\TexVision-Pro\data\TILDA_yolo\train\labels`
   - Click "Select Folder"

4. **Verify Classes**:
   - LabelImg should show these 4 classes:
     - hole
     - objects
     - oil spot
     - thread error

---

## 🎯 Annotation Process (Repeat for Each Image)

### For Each Image:

1. **View the Image**:
   - Look for fabric defects (holes, stains, objects, thread errors)

2. **Draw Bounding Box**:
   - Press `W` key (or click "Create RectBox")
   - Click and drag to draw a box around the defect
   - Make box tight around the defect (not too loose, not too tight)

3. **Select Class**:
   - Popup will appear with 4 classes
   - Select the correct defect type:
     - `hole` - Tears, holes in fabric
     - `objects` - Foreign objects on fabric
     - `oil spot` - Oil stains, spots
     - `thread error` - Thread defects, loose threads

4. **Save**:
   - Press `Ctrl+S` (or click "Save")
   - A .txt file will be created in the labels folder

5. **Next Image**:
   - Press `D` key (or click "Next Image")
   - Repeat steps 1-4

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `W` | Create bounding box |
| `D` | Next image |
| `A` | Previous image |
| `Ctrl+S` | Save |
| `Del` | Delete selected box |
| `Ctrl+D` | Duplicate box |

---

## 📊 Progress Tracking

### Validation Checkpoints

**Every 50 images**, validate your work:

```bash
python scripts/validate_annotations.py --split train
```

This will show:
- Number of annotated images
- Class distribution
- Any errors in format

### Target Progress

- **Minimum**: 300 images (train)
- **Recommended**: 500 images (train + val + test)
- **Optimal**: 800 images (all images)

**Current Progress**:
- Train: 5/480 (1%) ← START HERE
- Val: 0/160 (0%)
- Test: 0/160 (0%)

---

## ✅ Quality Guidelines

### Good Annotation:
- ✅ Box tightly fits the defect
- ✅ Entire defect is inside the box
- ✅ Minimal extra background
- ✅ Correct class label
- ✅ All defects in image are annotated

### Bad Annotation:
- ❌ Box too loose (too much background)
- ❌ Box too tight (cuts off defect)
- ❌ Wrong class label
- ❌ Missing defects in the image
- ❌ Overlapping boxes (unless defects overlap)

---

## 🚨 Common Mistakes to Avoid

1. **Wrong Format**: Using PascalVOC instead of YOLO
2. **Wrong Save Directory**: Saving to images folder instead of labels
3. **Missing Defects**: Not annotating all defects in an image
4. **Wrong Class**: Confusing similar defect types

---

## 📈 When to Stop and Train

### Minimum Viable Dataset
- **300 annotated images** in train set
- Run validation to check quality
- Proceed to training

### Recommended Dataset
- **480 annotated images** in train set (all train images)
- **100+ annotated images** in val set
- Better accuracy expected

---

## 🔄 After Annotation Complete

### Step 1: Validate All Annotations
```bash
python scripts/validate_annotations.py --data data/TILDA_yolo/data.yaml --plot
```

### Step 2: Start Training
```bash
python detection/yolo_trainer.py --config detection/configs/yolo_training.yaml --validate --export onnx
```

### Step 3: Monitor Training
```bash
tensorboard --logdir detection/runs/detect/train
```

### Step 4: Test Results
```bash
python detection/yolo_inference.py --model detection/runs/detect/train/weights/best.onnx --source test_image.png
```

---

## 💡 Tips for Faster Annotation

1. **Batch Similar Images**: Annotate all "hole" images together
2. **Use Keyboard Shortcuts**: `W`, `D`, `Ctrl+S` are your friends
3. **Take Breaks**: Every 50-100 images, take a 5-minute break
4. **Set Daily Goals**: Aim for 100 images per day
5. **Validate Often**: Check quality every 50 images

---

## ⏱️ Time Estimates

| Images | Time Required | Days (2hr/day) |
|--------|---------------|----------------|
| 100 | 2-3 hours | 1-2 days |
| 300 | 6-10 hours | 3-5 days |
| 500 | 10-16 hours | 5-8 days |
| 800 | 16-26 hours | 8-13 days |

---

## 🆘 Troubleshooting

### LabelImg Not Showing Classes
- Check if `classes.txt` exists in labels folder
- Should contain: hole, objects, oil spot, thread error

### Annotations Not Saving
- Verify save directory is set to `labels` folder
- Check write permissions
- Try manual save (`Ctrl+S`)

### Wrong Format
- Click the format button until it shows "YOLO"
- Delete any .xml files (those are PascalVOC format)

---

## ✨ You're Ready!

**LabelImg should be open now. Start annotating!**

Remember:
- Press `W` to draw boxes
- Press `Ctrl+S` to save
- Press `D` for next image
- Validate every 50 images

**Good luck! The training pipeline will run automatically once you reach 300+ annotations.**

---

**Need help?** See `docs/annotation_guide.md` for detailed instructions.
