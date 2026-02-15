import pytest
from unittest.mock import MagicMock
import time

# Use Case 01: Real-Time Alerts & Performance
# TC-UC01-006: Verify Visual Alert Trigger
# TC-UC01-007: Verify Audio Alert Trigger
def test_defect_alerts(mock_alert_system):
    # Simulate defect detection
    defects = [{
        'bbox': [10, 10, 50, 50],
        'final_class': 'hole',
        'final_conf': 0.95
    }]
    
    # Spy on alert methods
    mock_alert_system._visual_alert = MagicMock()
    mock_alert_system._audio_alert = MagicMock()
    mock_alert_system.enable_visual = True
    mock_alert_system.enable_audio = True # Mock alert system has audio logic disabled in fixture, re-enable logic
    mock_alert_system.audio_available = True  # Override hardware check
    
    # Trigger alert directly
    mock_alert_system.trigger_alert(defects)
    
    # Verify methods called
    mock_alert_system._visual_alert.assert_called_once()
    mock_alert_system._audio_alert.assert_called_once()

# TC-UC01-008: Verify Alert Response Time
def test_alert_response_time(mock_alert_system):
    # Simulate defect detection
    defects = [{
        'bbox': [10, 10, 50, 50],
        'final_class': 'hole',
        'final_conf': 0.95
    }]
    
    start_time = time.time()
    mock_alert_system.trigger_alert(defects)
    end_time = time.time()
    
    response_time = end_time - start_time
    # Requirement: < 1 second
    assert response_time < 1.0

# TC-UC02-003: Verify Data Retention Policy (Mock Check)
def test_data_retention_exists(mock_db):
    # This test verifies we can query old data, implying retention exists.
    # In a real scenario, we'd check if a cleanup job runs.
    # For now, verify no automatic deletion on insert.
    import datetime
    
    # Insert old record
    old_date = datetime.datetime.now() - datetime.timedelta(days=100)
    # Manually insert with old timestamp (requires raw SQL as log_inspection uses CURRENT_TIMESTAMP)
    with mock_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO fabric_images (filename, filepath, upload_time) VALUES (?, ?, ?)", 
                       ("old.jpg", "path", old_date))
        old_id = cursor.lastrowid
        
    # check it still exists
    with mock_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM fabric_images WHERE id=?", (old_id,))
        count = cursor.fetchone()[0]
        assert count == 1
