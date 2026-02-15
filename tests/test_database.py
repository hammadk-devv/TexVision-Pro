import pytest
import os
from datetime import datetime

# Tests for Defect Data Management
# TC-UC02-001: Verify Auto Save Defect Data
def test_auto_save_defect_data(mock_db, mocker):
    # Log an inspection
    filename = "test_fabric.jpg"
    filepath = "/tmp/test_fabric.jpg"
    defects = [{
        'bbox': [10, 10, 50, 50],
        'final_class': 'hole',
        'final_conf': 0.95
    }]
    
    image_id = mock_db.log_inspection(filename, filepath, defects)
    assert image_id is not None
    
    # Verify saved
    with mock_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inspection WHERE inspection_id=?", (image_id,))
        img_row = cursor.fetchone()
        assert os.path.basename(img_row[5]) == filename # image_path is at index 5
        
        cursor.execute("""
            SELECT d.defect_type, id.confidence 
            FROM inspection_defect id
            JOIN defect d ON id.defect_id = d.defect_id
            WHERE id.inspection_id=?
        """, (image_id,))
        defect_row = cursor.fetchone()
        assert defect_row is not None
        assert defect_row[0] == 'hole' # type
        assert defect_row[1] == 0.95   # confidence

# TC-UC02-002: Verify Data Integrity
def test_data_integrity(mock_db):
    # Insert specific data
    filename = "integrity_test.jpg"
    filepath = "/data/integrity_test.jpg"
    defects = [{
        'bbox': [100.5, 200.5, 300.5, 400.5],
        'final_class': 'oil spot',
        'final_conf': 0.887
    }]
    
    image_id = mock_db.log_inspection(filename, filepath, defects)
    
    # Retrieve and verify exact match
    with mock_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id.location_on_fabric, id.confidence, d.defect_type 
            FROM inspection_defect id
            JOIN defect d ON id.defect_id = d.defect_id
            WHERE id.inspection_id=?
        """, (image_id,))
        row = cursor.fetchone()
        
        import json
        bbox = json.loads(row[0])
        assert bbox == [100.5, 200.5, 300.5, 400.5]
        assert row[1] == 0.887
        assert row[2] == 'oil spot'

# TC-UC02-004: Verify Date Range Selection (via Reports)
def test_date_range_selection(report_gen, mock_db):
    # Insert data for today with at least one defect so it appears in report
    mock_db.log_inspection("today.jpg", "detections/today.jpg", [{'bbox':[0,0,10,10], 'final_class':'hole', 'final_conf':0.9}])
    
    # Generate CSV for today
    today = datetime.now().strftime('%Y-%m-%d')
    filepath, content = report_gen.generate_csv(date_str=today)
    
    assert "today.jpg" in content
    assert os.path.basename(filepath) == f"report_{today}.csv"

# TC-UC02-005: Verify PDF Report Generation
def test_pdf_report_generation(report_gen, mock_db):
    # Insert dummy data
    mock_db.log_inspection("pdf_test.jpg", "path", [{'bbox':[0,0,10,10], 'final_class':'hole', 'final_conf':0.9}])
    
    today = datetime.now().strftime('%Y-%m-%d')
    filepath = report_gen.generate_pdf(date_str=today)
    
    assert os.path.exists(filepath)
    assert filepath.endswith('.pdf')
    assert os.path.getsize(filepath) > 0

# TC-UC02-006: Verify CSV Report Generation
def test_csv_report_generation(report_gen, mock_db):
    # Insert dummy data
    mock_db.log_inspection("csv_test.jpg", "reports/csv_test.jpg", [{'bbox':[0,0,10,10], 'final_class':'hole', 'final_conf':0.9}])
    
    today = datetime.now().strftime('%Y-%m-%d')
    filepath, content = report_gen.generate_csv(date_str=today)
    
    assert os.path.exists(filepath)
    assert filepath.endswith('.csv')
    assert "ID,Filename,Timestamp" in content
    assert "csv_test.jpg" in content
