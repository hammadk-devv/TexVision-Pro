import cv2
import os

# Paths
imagePath = "data/TILDA_yolo/images/003_jpg.rf.7f120fb19741a8011c3aab5c414f9e3f.jpg"
labelPath = "data/TILDA_yolo/labels/003_jpg.rf.7f120fb19741a8011c3aab5c414f9e3f.txt"
outputPath = "outputs/debug_visualization.jpg"

os.makedirs("outputs", exist_ok=True)

# Read image
fabricImage = cv2.imread(imagePath)
imgH, imgW, _ = fabricImage.shape

# Read labels
with open(labelPath, "r") as f:
    labelLines = f.readlines()

for line in labelLines:
    parts = line.split()
    if len(parts) != 5: continue
    
    classId = parts[0]
    xC, yC, bW, bH = map(float, parts[1:])
    
    # Denormalize
    x1 = int((xC - bW/2) * imgW)
    y1 = int((yC - bH/2) * imgH)
    x2 = int((xC + bW/2) * imgW)
    y2 = int((yC + bH/2) * imgH)
    
    # Draw
    cv2.rectangle(fabricImage, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(fabricImage, classId, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Save
cv2.imwrite(outputPath, fabricImage)
print(f"Visualization saved to {outputPath}")
print(f"Image shape: {imgH}x{imgW}")
