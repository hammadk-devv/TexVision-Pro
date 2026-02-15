import sys
import os
import pytest
from unittest.mock import MagicMock
import sqlite3

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.deployment.database import TexVisionDB
from src.deployment.alert_system import AlertSystem
from src.deployment.reports import ReportGenerator

@pytest.fixture
def mock_db():
    """Create a temporary in-memory database for testing"""
    db = TexVisionDB(":memory:")
    return db

@pytest.fixture
def mock_detector(mocker):
    """Mock the IntegratedDetector"""
    detector = MagicMock()
    # Setup default mock return values
    detector.detect_and_classify.return_value = [
        {
            'bbox': [100, 100, 200, 200],
            'final_class': 'hole',
            'final_conf': 0.95
        }
    ]
    return detector

@pytest.fixture
def mock_alert_system(mocker):
    """Mock AlertSystem with audio disabled for tests"""
    alert_sys = AlertSystem()
    alert_sys.enable_audio = False  # Disable real audio
    alert_sys._visual_alert = MagicMock()
    alert_sys._audio_alert = MagicMock()
    alert_sys._log_alert = MagicMock()
    return alert_sys

@pytest.fixture
def report_gen(mock_db):
    """ReportGenerator fixture"""
    return ReportGenerator(mock_db)
