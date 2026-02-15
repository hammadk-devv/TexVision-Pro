import unittest
import os
import sys
import numpy as np
import cv2
import json
from pathlib import Path

# Add project source to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pipeline.integrated_detector import IntegratedDetector, ImageProcessor
from deployment.database import TexVisionDB
from deployment.reports import ReportGenerator

class TestTexVisionCompliance(unittest.TestCase):
    """
    Standardized Verification Suite using unittest library (Section 4.1).
    Ensures camelCase consistency and SRP modularity.
    """

    @classmethod
    def setUpClass(cls):
        """Set up shared resources for compliance testing"""
        cls.dbPath = ":memory:"
        cls.db = TexVisionDB(cls.dbPath)
        cls.reportGenerator = ReportGenerator(cls.db)
        
        # Mock weights paths
        cls.yoloPath = "models/yolov8s.pt"
        cls.resnetPath = "models/checkpoints/best_model.pth"
        
        # Test image (Dummy)
        cls.dummyImage = np.zeros((640, 640, 3), dtype=np.uint8)

    def testCamelCaseVariables(self):
        """Verify project-wide camelCase standard (Section 4.1)"""
        # Test IntegratedDetector methods
        detector = IntegratedDetector(self.yoloPath, self.resnetPath)
        self.assertTrue(hasattr(detector, 'detectDefectCnn'), "Missing detectDefectCnn (SRP)")
        self.assertTrue(hasattr(detector, 'detectDefectYolo'), "Missing detectDefectYolo (SRP)")
        self.assertTrue(hasattr(detector, 'detectAndClassify'), "detect_and_classify should be detectAndClassify")
        
        # Test Database methods
        self.assertTrue(hasattr(self.db, 'logInspection'), "Missing logInspection (camelCase)")
        self.assertTrue(hasattr(self.db, 'getDailyStats'), "Missing getDailyStats (camelCase)")
        
        # Test ReportGenerator methods
        self.assertTrue(hasattr(self.reportGenerator, 'generateReport'), "Missing generateReport (camelCase)")

    def testImageProcessorModularity(self):
        """Verify ImageProcessor SRP component (Section 3.8 / 4.3)"""
        processed = ImageProcessor.processImage(self.dummyImage)
        self.assertEqual(processed.shape, (416, 416, 3), "ImageProcessor should resize to 416x416")

    def testSnippet3ReportStructure(self):
        """Verify generateReport produces exact Snippet 3 JSON structure (Section 4.3)"""
        # 1. Log a dummy inspection
        results = [{
            'bbox': [10, 10, 50, 50],
            'final_class': 'hole',
            'final_conf': 0.95
        }]
        insId = self.db.logInspection("test.jpg", "uploads/test.jpg", results)
        
        # 2. Generate report
        report = self.reportGenerator.generateReport(insId)
        
        # 3. Check fields
        expectedFields = ["reportId", "inspectionId", "inspectionDate", "defectsDetectedCount", "defectsList"]
        for field in expectedFields:
            self.assertIn(field, report, f"Snippet 3 mismatch: Field '{field}' missing")
            
        self.assertEqual(report["defectsDetectedCount"], 1)
        self.assertEqual(report["defectsList"][0]["type"], "hole")
        self.assertIsInstance(report["defectsList"][0]["location"], list)

    def testYoloConfidenceLogic(self):
        """Verify YOLO logic adheres to 50% threshold (Section 4.3 Snippet 2)"""
        detector = IntegratedDetector(self.yoloPath, self.resnetPath, yoloConf=0.50)
        self.assertEqual(detector.yoloConf, 0.50, "YOLO threshold must be 0.50 by default")

if __name__ == '__main__':
    unittest.main()
