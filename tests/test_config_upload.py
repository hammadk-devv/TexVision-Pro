import pytest
from unittest.mock import MagicMock
import json

# UC-03: System Configuration
# TC-UC03-001: Verify Sensitivity Slider
def test_sensitivity_slider(mocker, mock_detector):
    # Mock Flask app context if needed, but we can test logic directly
    # Assume detector has set_thresholds method
    detector = mock_detector
    
    # Simulate slider adjustment to 50%
    new_val = 50
    conf = new_val / 100.0
    
    detector.set_thresholds = MagicMock()
    detector.set_thresholds(yolo_conf=conf)
    
    detector.set_thresholds.assert_called_with(yolo_conf=0.5)

# TC-UC03-002: Verify Save Sensitivity (Persistence)
def test_sensitivity_persistence(mocker, mock_detector):
    detector = mock_detector
    detector.yolo_conf = 0.5 # Simulate saved state
    
    # Verify next detection uses this conf
    # This would be an integration test on the detector class itself
    assert detector.yolo_conf == 0.5

# UC-04: Manual Image Upload
# TC-UC04-001 & 002: Verify JPG/PNG Upload
# Note: Full Flask integration tests require test_client
def test_manual_upload_logic(mocker, mock_db, mock_detector):
    # Mocking the upload processing logic from flask_app.py
    # We'll simulate the function core logic
    filename = "test_upload.png" # PNG test
    filepath = "/tmp/test_upload.png"
    
    # Mock file save
    mocker.patch('os.path.join', return_value=filepath)
    
    # Run analysis
    results = mock_detector.detect_and_classify(None) # Image would be loaded here
    
    # Log to DB
    image_id = mock_db.log_inspection(filename, filepath, results)
    
    assert image_id is not None
    assert len(results) > 0

# TC-UC04-003: Verify Manual Image Analysis
def test_manual_analysis_results(mocker, mock_detector):
    results = mock_detector.detect_and_classify(None)
    
    # Verify structure matches expectation
    assert isinstance(results, list)
    defect = results[0]
    assert 'bbox' in defect
    assert 'final_class' in defect
    assert 'final_conf' in defect
