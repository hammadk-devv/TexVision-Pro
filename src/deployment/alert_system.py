"""
Alert System for Defect Detection
Refactored for Section 4.1 Coding Standards and Hardware Signaling
"""

import time
from typing import List, Dict
from datetime import datetime
import json
from pathlib import Path
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

class AlertSystem:
    """
    Industrial alert manager for TexVision-Pro.
    Supports console, audio, and hardware signals (Red/Green LEDs).
    """
    
    def __init__(
        self,
        enableAudio: bool = True,
        enableVisual: bool = True,
        enableLogging: bool = True,
        logFile: str = 'defect_log.json',
        cooldownSeconds: float = 2.0
    ):
        """
        Initialize the multi-modal alert system.

        :param enableAudio: Toggle sound notifications
        :param enableVisual: Toggle console/UI indicators
        :param enableLogging: Toggle disk logging
        :param logFile: Output JSON path
        :param cooldownSeconds: Threshold between consecutive alerts
        """
        self.enableAudio = enableAudio
        self.enableVisual = enableVisual
        self.enableLogging = enableLogging
        self.logFile = Path(logFile)
        self.cooldownSeconds = cooldownSeconds
        
        # Hardware interfaces (Architecture 3.2.2)
        self.hardware = RPiHardwareAlert()
        self.arduino = ArduinoAlert()
        
        self.lastAlertTime = 0
        self.alertCount = 0
        
        # Audio Engine Initializer
        if self.enableAudio:
            try:
                import pygame
                pygame.mixer.init()
                self.audioAvailable = True
            except ImportError:
                self.audioAvailable = False
        else:
            self.audioAvailable = False
        
        if self.enableLogging and not self.logFile.exists():
            self.logFile.write_text('[]')
    
    def triggerAlert(self, detections: List[Dict]):
        """
        Orchestrate alerts across all enabled channels.

        :param detections: Result list from IntegratedDetector
        """
        currentTime = time.time()
        if currentTime - self.lastAlertTime < self.cooldownSeconds:
            return
        
        self.lastAlertTime = currentTime
        
        # Filter significant defects
        realDefects = [d for d in detections if d['final_class'] != 'false_positive']
        if not realDefects:
            return
            
        self.alertCount += 1
        
        if self.enableVisual: self._visualAlert(realDefects)
        if self.enableAudio and self.audioAvailable: self._audioAlert(realDefects)
        if self.enableLogging: self._logAlert(realDefects)
            
        # Hardware Integration (Section 3.2.2)
        self.hardware.triggerDefectAlert(len(realDefects))
        self.arduino.sendAlert(len(realDefects))
    
    def _visualAlert(self, defects: List[Dict]):
        """Internal console notification."""
        print(f"\n[!] DEFECT DETECTED at {datetime.now().strftime('%H:%M:%S')}")
        for i, d in enumerate(defects, 1):
            print(f"  ({i}) {d['final_class']} | Conf: {d['final_conf']:.2f}")
    
    def _audioAlert(self, defects: List[Dict]):
        """Internal audio signal trigger."""
        print("🔊 [Audio Beep Triggered]")
    
    def _logAlert(self, defects: List[Dict]):
        """Disk logging (Section 3.6.4)."""
        try:
            logData = json.loads(self.logFile.read_text()) if self.logFile.exists() else []
            entry = {
                'timestamp': datetime.now().isoformat(),
                'alertId': self.alertCount,
                'defectCount': len(defects),
                'details': defects
            }
            logData.append(entry)
            self.logFile.write_text(json.dumps(logData, indent=2))
        except Exception as e:
            print(f"Log failure: {e}")
    
    def triggerCameraAlert(self):
        """E1-US-1: Dedicated alert for sensor disconnection."""
        print("⚠️  SYSTEM ALERT: CAMERA DISCONNECTED")
        if self.enableAudio and self.audioAvailable:
            self._audioAlert([])
    
    def getStatistics(self) -> Dict:
        """
        Retrieve session alert metrics.
        :return: Metric summary dict
        """
        return {
            'totalAlerts': self.alertCount,
            'logPath': str(self.logFile) if self.enableLogging else None
        }

class RPiHardwareAlert:
    """
    GPIO Interface for Raspberry Pi 4 (Hardware Spec 3.3).
    """
    def __init__(self, redPin: int = 17, greenPin: int = 27, buzzerPin: int = 22):
        self.redPin = redPin
        self.greenPin = greenPin
        self.buzzerPin = buzzerPin
        self.gpioAvailable = False
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup([self.redPin, self.greenPin, self.buzzerPin], GPIO.OUT)
            GPIO.output(self.greenPin, GPIO.HIGH) # Ready
            self.gpioAvailable = True
        except:
            pass
    
    def triggerDefectAlert(self, count: int, duration: float = 1.0):
        """Toggle hardware indicators."""
        if not self.gpioAvailable: return
        self.GPIO.output(self.greenPin, self.GPIO.LOW)
        self.GPIO.output(self.redPin, self.GPIO.HIGH)
        self.GPIO.output(self.buzzerPin, self.GPIO.HIGH)
        time.sleep(duration)
        self.GPIO.output(self.redPin, self.GPIO.LOW)
        self.GPIO.output(self.buzzerPin, self.GPIO.LOW)
        self.GPIO.output(self.greenPin, self.GPIO.HIGH)

class ArduinoAlert:
    """
    Serial Interface for Arduino Markers (Architecture 3.2.2).
    """
    def __init__(self, port=None, baudrate=9600):
        self.ser = None
        if SERIAL_AVAILABLE:
            if not port:
                ports = list(serial.tools.list_ports.comports())
                for p in ports:
                    if 'Arduino' in p.description: port = p.device; break
            if port:
                try:
                    self.ser = serial.Serial(port, baudrate, timeout=1)
                except:
                    pass

    def sendAlert(self, defectCount: int):
        """Write defect count byte to serial bus."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(bytes([min(defectCount, 255)]))
            except:
                pass
