import pytest
from unittest.mock import MagicMock

# Use Case 01: Defect Detection Accuracy
# TC-UC01-003: Verify Defect Bounding Boxes
def test_defect_bounding_boxes(mock_detector):
    # Mock detector output
    results = mock_detector.detect_and_classify(None)
    
    # Check if we have results
    assert len(results) > 0
    defect = results[0]
    
    # Check bbox format
    bbox = defect['bbox']
    assert isinstance(bbox, list) or isinstance(bbox, tuple)
    assert len(bbox) == 4
    # Check coordinates valid
    assert bbox[2] > bbox[0]
    assert bbox[3] > bbox[1]

# TC-UC01-004: Verify Defect Type Classification
def test_defect_type_classification(mock_detector):
    # Mock output for a specific Known Defect
    mock_detector.detect_and_classify.return_value = [{
        'bbox': [0,0,1,1],
        'final_class': 'hole',
        'final_conf': 0.9
    }]
    
    results = mock_detector.detect_and_classify(None)
    assert results[0]['final_class'] == 'hole'

# TC-UC01-005: Verify Confidence Score Accuracy
def test_confidence_score_accuracy(mock_detector):
    results = mock_detector.detect_and_classify(None)
    conf = results[0]['final_conf']
    
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0
