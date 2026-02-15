"""
Real-time Camera Processor for TexVision-Pro
Processes camera feed and detects defects in real-time.
Refactored for Section 4.1 Coding Standards and 1080p Calibration.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
import argparse
from typing import List, Dict, Tuple, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.integrated_detector import IntegratedDetector, ImageProcessor
from deployment.alert_system import AlertSystem
from deployment.database import AsyncDatabaseBridge

class CameraProcessor:
    """
    Real-time multi-threaded camera feed processor.
    Optimized for 1080p acquisition and 416x416 analysis (Section 3.7).
    """
    
    def __init__(
        self,
        yoloModelPath: str,
        resnetModelPath: str,
        cameraId: int = 0,
        frameSkip: int = 2,
        display: bool = True,
        saveDetections: bool = True,
        saveDir: str = 'detections',
        device: str = 'cpu'
    ):
        """
        Initialize the camera processor module.

        :param yoloModelPath: Path to YOLO weights
        :param resnetModelPath: Path to Keras ResNet weights
        :param cameraId: Hardware index for the image sensor
        :param frameSkip: Analysis frequency (skip N frames)
        :param display: Toggle video window
        :param saveDetections: Toggle disk saving of defect frames
        :param saveDir: Storage directory for frames
        :param device: Inference device ('cpu' or 'cuda')
        """
        self.cameraId = cameraId
        self.frameSkip = frameSkip
        self.display = display
        self.saveDetections = saveDetections
        self.saveDir = Path(saveDir)
        self.device = device
        
        if self.saveDetections:
            self.saveDir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Sub-Components (Section 4.3 SRP)
        self.detector = IntegratedDetector(
            yoloModelPath=yoloModelPath,
            resnetModelPath=resnetModelPath,
            device=device
        )
        self.alertSystem = AlertSystem()
        self.dbBridge = AsyncDatabaseBridge()
        
        # Initialize Camera sensor (Hardware Spec Figure 3.3)
        self.cap = cv2.VideoCapture(cameraId)
        if not self.cap.isOpened():
            raise RuntimeError(f"Sensor failure at ID {cameraId}")
        
        # Standardize Acquisition to 1080p (Section 3.3)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.frameCount = 0
        self.processedCount = 0
        self.defectCount = 0
        self.startTime = time.time()

    def processFrame(self, frame: np.ndarray) -> tuple:
        """
        Execute the localized analysis pipeline (Section 3.6).

        :param frame: Raw 1080p frame
        :return: (Annotated Frame, Results List, Has Defects Flag)
        """
        # Step 4: Preprocessing (ImageProcessor Figure 3.3)
        processedImage = ImageProcessor.processImage(frame)
        
        # Step 5: Detection Pipeline (IntegratedDetector SRP)
        analysisResults = self.detector.detectAndClassify(processedImage)
        
        # Cleanup class_ids for alerting
        hasDefects = len(analysisResults) > 0
        
        return analysisResults, hasDefects

    def run(self):
        """Infinite loop for real-time acquisition and monitoring."""
        print(f"🚀 Monitoring started on Camera {self.cameraId}")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.alertSystem.triggerCameraAlert()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.cameraId)
                    continue
                
                self.frameCount += 1
                if self.frameCount % self.frameSkip == 0:
                    results, hasDefects = self.processFrame(frame)
                    self.processedCount += 1
                    
                    if hasDefects:
                        self.defectCount += 1
                        self.alertSystem.triggerAlert(results)
                        
                        # Background data sync (Architecture 3.2.2)
                        if self.saveDetections:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            frameName = f"defect_{timestamp}.jpg"
                            savePath = self.saveDir / frameName
                            cv2.imwrite(str(savePath), frame)
                            self.dbBridge.logInspectionAsync(
                                filename=frameName,
                                filepath=str(savePath),
                                defectsList=results
                            )
                
                if self.display:
                    cv2.imshow('TexVision-Pro Sensor Feed', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.dbBridge.stop()

def main():
    parser = argparse.ArgumentParser(description='TexVision-Pro 1080p Acquisition')
    parser.add_argument('--yolo', type=str, default='models/yolov8s.pt')
    parser.add_argument('--resnet', type=str, default='models/checkpoints/best_model.pth')
    args = parser.parse_args()
    
    processor = CameraProcessor(yoloModelPath=args.yolo, resnetModelPath=args.resnet)
    processor.run()

if __name__ == "__main__":
    main()
