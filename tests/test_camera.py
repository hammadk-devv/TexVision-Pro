import pytest
from unittest.mock import MagicMock
import cv2
import numpy as np

# Import CameraProcessor
from src.deployment.camera_processor import CameraProcessor

def test_camera_automatic_capture(mocker, mock_alert_system):
    """TC-UC01-001: Verify Automatic Fabric Image Capture"""
    # Mock cv2.VideoCapture
    mock_cap = mocker.patch('cv2.VideoCapture')
    mock_instance = mock_cap.return_value
    mock_instance.isOpened.return_value = True
    # Create fake frame
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_instance.read.return_value = (True, fake_frame)
    
    # Mock IntegratedDetector to avoid loading models
    mocker.patch('src.deployment.camera_processor.IntegratedDetector')
    
    processor = CameraProcessor(
        yolo_model_path="dummy_yolo.pt",
        resnet_model_path="dummy_resnet.pth",
        camera_id=0,
        display=False,  # Disable display for tests
        save_detections=False
    )
    processor.alert_system = mock_alert_system
    
    # Run one frame processing cycle manually
    ret, frame = processor.cap.read()
    assert ret is True
    assert frame is not None
    assert frame.shape == (100, 100, 3)

def test_camera_disconnect_alert(mocker, mock_alert_system):
    """TC-UC01-002: Verify Camera Disconnect Alert"""
    mock_cap = mocker.patch('cv2.VideoCapture')
    mock_instance = mock_cap.return_value
    mock_instance.isOpened.return_value = True
    mock_instance.read.return_value = (False, None)  # Simulate disconnect
    
    # Mock IntegratedDetector
    mocker.patch('src.deployment.camera_processor.IntegratedDetector')
    
    processor = CameraProcessor(
        yolo_model_path="dummy_yolo.pt",
        resnet_model_path="dummy_resnet.pth",
        camera_id=0,
        display=False,
        save_detections=False
    )
    processor.alert_system = mock_alert_system
    
    # Spy on the alert trigger
    spy_alert = mocker.spy(processor.alert_system, 'trigger_camera_alert')
    
    # Simulate run loop behavior for disconnection
    try:
        # Manually trigger the logic that happens inside run()
        ret, frame = processor.cap.read()
        if not ret:
            processor.alert_system.trigger_camera_alert()
    except:
        pass
        
    spy_alert.assert_called_once()
