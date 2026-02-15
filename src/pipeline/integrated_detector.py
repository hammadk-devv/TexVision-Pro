"""
Integrated Defect Detection Pipeline
Refactored for Section 4.1 Coding Standards and Section 4.3 Software Description
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import sys
import json
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from detection.yolo_inference import YOLODetector

class ImageProcessor:
    """
    Reflects Figure 3.3: ImageProcessor Component (Section 3.8)
    """
    @staticmethod
    def processImage(image: np.ndarray, targetSize=(416, 416)) -> np.ndarray:
        """
        Preprocessing tasks: resizing and contrast enhancement (Section 3.7)

        :param image: Input fabric image (BGR)
        :param targetSize: Target dimensions for YOLO
        :return: Preprocessed and resized image
        """
        # Contrast Enhancement (OpenCV Section 3.7)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Resize for YOLO (Architecture requirement)
        return cv2.resize(enhanced, targetSize)

class IntegratedDetector:
    """
    Acts as the DefectDetectionModule (Figure 3.3 / Section 4.3)
    
    Adheres to modularity standards (SRP) by separating 
    classification and localization tasks.
    """
    
    def __init__(
        self,
        yoloModelPath: str,
        resnetModelPath: str,
        yoloConf: float = 0.50, # Section 4.3: applies threshold of 50%
        yoloIou: float = 0.45,
        resnetConf: float = 0.80,
        cropPadding: int = 20,
        minDefectSize: int = 32,
        device: str = 'cpu'
    ):
        """
        Initialize the detector module with Keras and YOLO backends.

        :param yoloModelPath: Path to YOLO weights
        :param resnetModelPath: Path to Keras ResNet-50 weights
        :param yoloConf: Localization confidence threshold (Standard: 0.5)
        :param yoloIou: IoU threshold for overlapping defects
        :param resnetConf: ResNet classification confidence threshold
        :param cropPadding: Pixel padding for classification crops
        :param minDefectSize: Minimum size to process a defect region
        :param device: Inference device ('cpu' or 'cuda')
        """
        self.yoloConf = yoloConf
        self.yoloIou = yoloIou
        self.resnetConf = resnetConf
        self.cropPadding = cropPadding
        self.minDefectSize = minDefectSize
        self.device = device
        
        # Initialize YOLO detector (DefectDetectionModule abstraction)
        print("Loading YOLO detector...")
        self.yoloDetector = YOLODetector(
            model_path=yoloModelPath,
            conf_threshold=yoloConf,
            iou_threshold=yoloIou,
            device=device
        )
        
        # Initialize Keras ResNet classifier (Section 3.7)
        print("Loading Keras ResNet classifier...")
        self.resnetModel = self._loadResNet(resnetModelPath)
        
        # ResNet class names (Standardized naming)
        self.resnetClasses = ['good', 'hole', 'objects', 'oil spot', 'thread error']
        
        # Performance metrics
        self.totalTime = 0
        self.yoloTime = 0
        self.resnetTime = 0
        self.numProcessed = 0

    def setThresholds(self, yoloConf=None, resnetConf=None):
        """
        Dynamically update confidence thresholds.

        :param yoloConf: New YOLO confidence limit
        :param resnetConf: New ResNet confidence limit
        """
        if yoloConf is not None:
            self.yoloConf = yoloConf
            self.yoloDetector.conf_threshold = yoloConf
        if resnetConf is not None:
            self.resnetConf = resnetConf
        
    def _loadResNet(self, modelPath: str):
        """
        Load Keras ResNet-50 model (Section 3.7)

        :param modelPath: Disk path to weights
        :return: Loaded Keras model
        """
        baseModel = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(baseModel.output)
        output = tf.keras.layers.Dense(5, activation='softmax')(x)
        model = tf.keras.Model(inputs=baseModel.input, outputs=output)
        
        if modelPath.endswith(('.h5', '.keras')):
            try:
                model.load_weights(modelPath)
            except Exception as e:
                 print(f"⚠️ Warning: Keras weights load failure: {e}")
        return model
    
    def detectDefectCnn(self, fabricImage: np.ndarray) -> Tuple[str, float]:
        """
        SRP Component: Classification Using CNN (Section 4.3 / Snippet 1)
        Outputs labels: 'Defective' or 'Non-Defective'.

        :param fabricImage: Input image for global check
        :return: (Status Label, Confidence Score)
        """
        # Preprocess for Keras ResNet (Section 3.6.2)
        imgRgb = cv2.cvtColor(fabricImage, cv2.COLOR_BGR2RGB)
        imgResized = cv2.resize(imgRgb, (224, 224))
        imgArray = tf.keras.utils.img_to_array(imgResized)
        imgBatch = np.expand_dims(imgArray, axis=0)
        imgPreprocessed = preprocess_input(imgBatch)
        
        # Inference (Section 4.3)
        predictions = self.resnetModel.predict(imgPreprocessed, verbose=0)
        confidence = np.max(predictions[0])
        classId = np.argmax(predictions[0])
        
        label = "Defective" if classId > 0 else "Non-Defective"
        return label, confidence

    def detectDefectYolo(self, fabricImage: np.ndarray) -> List[Dict]:
        """
        SRP Component: Localization Using YOLO (Section 4.3 / Snippet 2)
        returns coordinates and confidence levels.

        :param fabricImage: Image for defect localization
        :return: List of detected objects with box [x,y,w,h] and class_id
        """
        # Note: This logic follows Snippet 2's structure
        height, width = fabricImage.shape[:2]
        
        # Call underlying YOLO detector
        # We wrap the existing implementation to match Section 4.3 Snippet 2 interface/naming
        yoloResults, crops = self.yoloDetector.detect(fabricImage, return_crops=True)
        
        detectedObjects = []
        for det in yoloResults:
            # Coordinates from yolo_inference are [x1, y1, x2, y2]
            x1, y1, x2, y2 = det['bbox']
            
            # Convert to [x, y, w, h] as per Snippet 2
            w = x2 - x1
            h = y2 - y1
            
            # Validation Step matches Snippet 2 Logic
            if det['confidence'] > 0.5: # 50% threshold enforcement
                detectedObjects.append({
                    "class_id": det['class_id'],
                    "confidence": det['confidence'],
                    "box": [int(x1), int(y1), int(w), int(h)],
                    "crop": crops[yoloResults.index(det)] # Internal helper for secondary classification
                })
        
        return detectedObjects

    def detectAndClassify(self, fabricImage: np.ndarray) -> List[Dict]:
        """
        Full integrated pipeline orchestrator.

        :param fabricImage: Processed image from ImageProcessor
        :return: Final aggregated results list
        """
        startTime = time.time()
        
        # Step 1: SRP CNN Check (Section 4.3)
        cnnLabel, cnnConf = self.detectDefectCnn(fabricImage)
        
        if cnnLabel == "Non-Defective":
            self.numProcessed += 1
            self.totalTime += time.time() - startTime
            return []
            
        # Step 2: SRP YOLO Localization (Section 4.3)
        detectedObjects = self.detectDefectYolo(fabricImage)
        
        results = []
        for obj in detectedObjects:
            # Step 3: Secondary Classification for precise type (Section 3.6.3)
            # Preprocess crop
            cropRgb = cv2.cvtColor(obj['crop'], cv2.COLOR_BGR2RGB)
            cropResized = cv2.resize(cropRgb, (224, 224))
            cropArray = tf.keras.utils.img_to_array(cropResized)
            cropBatch = np.expand_dims(cropArray, axis=0)
            cropPreprocessed = preprocess_input(cropBatch)
            
            # Predict
            preds = self.resnetModel.predict(cropPreprocessed, verbose=0)
            resId = np.argmax(preds[0])
            resConf = np.max(preds[0])
            resName = self.resnetClasses[resId]
            
            # Map back to result format
            x, y, w, h = obj['box']
            results.append({
                'bbox': [x, y, x + w, y + h],
                'yolo_class': 'defect', # Base YOLO class
                'yolo_conf': float(obj['confidence']),
                'resnet_class': resName,
                'resnet_conf': float(resConf),
                'final_class': resName if resName != 'good' else 'defect',
                'final_conf': float(min(obj['confidence'], resConf))
            })
            
        self.numProcessed += 1
        self.totalTime += time.time() - startTime
        return results

    def visualize(self, image: np.ndarray, results: List[Dict], savePath: str = None) -> np.ndarray:
        """
        Overlay defect regions on image (Section 4.3).

        :param image: Fabric image
        :param results: Analysis outcome
        :param savePath: Optional output file path
        :return: Annotated image
        """
        annotated = image.copy()
        for res in results:
            x1, y1, x2, y2 = map(int, res['bbox'])
            # Contrasting hue (Red for defect Section 4.3)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f"{res['final_class']} {res['final_conf']:.2f}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if savePath:
            cv2.imwrite(savePath, annotated)
        return annotated

    def setThresholds(self, yoloConf: float = None, resnetConf: float = None):
        """
        Dynamically update detection sensitivity (Section 4.3).

        :param yoloConf: New YOLO confidence threshold
        :param resnetConf: New ResNet confidence threshold
        """
        if yoloConf is not None:
            self.yoloConf = yoloConf
            # Update underlying detector
            self.yoloDetector.conf_threshold = yoloConf
        if resnetConf is not None:
            self.resnetConf = resnetConf

    def getPerformanceStats(self) -> Dict:
        """
        Retrieve timing metrics.
        
        :return: Stats dict
        """
        if self.numProcessed == 0: return {}
        avgTotal = self.totalTime / self.numProcessed
        return {
            'avgTotalTime': avgTotal,
            'avgFps': 1.0 / avgTotal if avgTotal > 0 else 0,
            'numProcessed': self.numProcessed
        }
