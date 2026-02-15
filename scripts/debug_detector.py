import cv2
import sys
import os
from pathlib import Path

# Add root to sys.path
sys.path.append(os.getcwd())

from src.detection.yolo_inference import YOLODetector
from src.pipeline.integrated_detector import IntegratedDetector

def test_on_real_data():
    model_path = r"runs/detect/train3/weights/best.onnx"
    image_path = r"data/TILDA_yolo/images/000_jpg.rf.37110a5377a77cf7b46daac60c0158cd.jpg"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        # Try PT instead
        model_path = r"runs/detect/train3/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"PT Model not found either: {model_path}")
            return

    print(f"Using model: {model_path}")
    resnet_path = r"checkpoints/best_model.pth"
    detector = IntegratedDetector(
        yolo_model_path=model_path, 
        resnet_model_path=resnet_path,
        min_defect_size=10
    )    
    # Set lower threshold to see if we find ANYTHING
    detector.set_thresholds(yolo_conf=0.1, resnet_conf=0.1)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return
        
    results = detector.detect_and_classify(image)
    print(f"Detected {len(results)} defects")
    for i, res in enumerate(results):
        print(f"Defect {i+1}: {res['type']} at {res['bbox']} (Conf: {res['confidence']:.4f})")
    
    # Save visualization
    vis_image = detector.visualize(image, results)
    cv2.imwrite("logs/debug_test_000.jpg", vis_image)
    print(f"Visualization saved to logs/debug_test_000.jpg")

if __name__ == "__main__":
    test_on_real_data()
