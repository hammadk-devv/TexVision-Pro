import sqlite3
import os
from datetime import datetime
from pathlib import Path
import threading
import queue
import time
import json

class TexVisionDB:
    """
    Relational Database Management for TexVision-Pro.
    Aligned with Section 3.5 Schema and Section 4.1 Coding Standards.
    """
    def __init__(self, dbPath="data/texvision.db"):
        """
        Initialize database connection.
        :param dbPath: Path to SQLite file
        """
        self.dbPath = dbPath
        self.sharedConn = None
        if self.dbPath == ":memory:":
            self.sharedConn = sqlite3.connect(self.dbPath)
        self.initDb()

    def getConnection(self):
        """
        Get SQLite connection.
        :return: sqlite3 connection object
        """
        if self.sharedConn:
            return self.sharedConn
        return sqlite3.connect(self.dbPath)

    def initDb(self):
        """
        Initialize database with normalized schema (Section 3.5.2).
        Uses snake_case for physical columns but camelCase for logic.
        """
        with self.getConnection() as conn:
            cursor = conn.cursor()
            
            # 1. Fabric table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fabric (
                    fabric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fabric_type TEXT NOT NULL,
                    batch_number TEXT,
                    manufacture_date TIMESTAMP
                )
            ''')
            
            # 2. Inspector table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inspector (
                    inspector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    qualification TEXT,
                    password TEXT,
                    role TEXT
                )
            ''')
            
            # 3. Inspection table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inspection (
                    inspection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspector_id INTEGER NOT NULL,
                    fabric_id INTEGER,
                    inspection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    inspection_notes TEXT,
                    image_path TEXT,
                    FOREIGN KEY (inspector_id) REFERENCES inspector (inspector_id),
                    FOREIGN KEY (fabric_id) REFERENCES fabric (fabric_id)
                )
            ''')
            
            # 4. Defect table (Catalog)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS defect (
                    defect_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    defect_type TEXT UNIQUE NOT NULL,
                    severity_level TEXT
                )
            ''')
            
            # 5. Inspection_Defect (Junction)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inspection_defect (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id INTEGER NOT NULL,
                    defect_id INTEGER NOT NULL,
                    defect_description TEXT,
                    location_on_fabric TEXT,
                    confidence REAL,
                    FOREIGN KEY (inspection_id) REFERENCES inspection (inspection_id),
                    FOREIGN KEY (defect_id) REFERENCES defect (defect_id)
                )
            ''')
            
            # 6. Report table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS report (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id INTEGER NOT NULL,
                    report_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary TEXT,
                    FOREIGN KEY (inspection_id) REFERENCES inspection (inspection_id)
                )
            ''')

            # Seed initial catalog data
            cursor.execute("INSERT OR IGNORE INTO defect (defect_type, severity_level) VALUES (?, ?)", ('hole', 'Critical'))
            cursor.execute("INSERT OR IGNORE INTO defect (defect_type, severity_level) VALUES (?, ?)", ('oil spot', 'Medium'))
            cursor.execute("INSERT OR IGNORE INTO defect (defect_type, severity_level) VALUES (?, ?)", ('objects', 'Medium'))
            cursor.execute("INSERT OR IGNORE INTO defect (defect_type, severity_level) VALUES (?, ?)", ('thread error', 'Low'))
            
            # Seed default admin as inspector
            cursor.execute("SELECT count(*) FROM inspector WHERE name = 'admin'")
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO inspector (name, password, role, qualification) VALUES (?, ?, ?, ?)",
                    ('admin', 'admin123', 'admin', 'Senior Quality Auditor')
                )
                
            conn.commit()

    def logInspection(self, filename: str, filepath: str, defectsList: list, userId: int = 1) -> int:
        """
        Log normalized inspection event (Section 3.6.4).
        
        :param filename: Original image file name
        :param filepath: Path to saved image
        :param defectsList: List of detection dictionaries
        :param userId: ID of the responsible inspector
        :return: Generated Inspection ID
        """
        with self.getConnection() as conn:
            cursor = conn.cursor()
            
            # 1. Ensure a Fabric record exists
            cursor.execute("SELECT fabric_id FROM fabric LIMIT 1")
            fabricRow = cursor.fetchone()
            if not fabricRow:
                cursor.execute(
                    "INSERT INTO fabric (fabric_type, batch_number) VALUES (?, ?)",
                    ('Industrial Fabric', 'BATCH-001')
                )
                fabricId = cursor.lastrowid
            else:
                fabricId = fabricRow[0]
            
            # 2. Insert Inspection
            cursor.execute(
                "INSERT INTO inspection (inspector_id, fabric_id, image_path) VALUES (?, ?, ?)",
                (userId, fabricId, filepath)
            )
            inspectionId = cursor.lastrowid
            
            # 3. Insert Defects into Junction Table
            for det in defectsList:
                cursor.execute("SELECT defect_id FROM defect WHERE defect_type = ?", (det['final_class'],))
                defRow = cursor.fetchone()
                if defRow:
                    defectId = defRow[0]
                else:
                    cursor.execute("INSERT INTO defect (defect_type, severity_level) VALUES (?, ?)", (det['final_class'], 'Medium'))
                    defectId = cursor.lastrowid
                
                bboxStr = json.dumps(det.get('bbox', [0, 0, 0, 0]))
                
                cursor.execute(
                    """INSERT INTO inspection_defect (inspection_id, defect_id, defect_description, location_on_fabric, confidence) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (inspectionId, defectId, f"Detected {det['final_class']}", bboxStr, det['final_conf'])
                )
            
            # 4. Create an optional summary Report
            if len(defectsList) > 0:
                summary = f"Identified {len(defectsList)} defects."
                cursor.execute(
                    "INSERT INTO report (inspection_id, summary) VALUES (?, ?)",
                    (inspectionId, summary)
                )
            
            conn.commit()
            return inspectionId

    def getDailyStats(self, dateStr: str = None) -> dict:
        """
        Retrieve performance statistics for a specific date.
        
        :param dateStr: Target date (YYYY-MM-DD or None for today)
        :return: Stats dictionary
        """
        if dateStr is None:
            dateStr = datetime.now().strftime('%Y-%m-%d')
            
        with self.getConnection() as conn:
            cursor = conn.cursor()
            
            # Total Inspections
            cursor.execute(
                "SELECT COUNT(*) FROM inspection WHERE date(inspection_date) = ?", 
                (dateStr,)
            )
            totalCount = cursor.fetchone()[0]
            
            # Total Defects found
            cursor.execute(
                """SELECT COUNT(*) FROM inspection_defect 
                   JOIN inspection ON inspection_defect.inspection_id = inspection.inspection_id 
                   WHERE date(inspection.inspection_date) = ?""", 
                (dateStr,)
            )
            defectCount = cursor.fetchone()[0]
            
            passRate = 100.0 if totalCount == 0 else ((totalCount - defectCount) / totalCount) * 100.0
            
            return {
                'total_inspections': totalCount,
                'defects_found': defectCount,
                'pass_rate': round(max(0, passRate), 1)
            }

    def getRecentInspections(self, limit: int = 10) -> list:
        """
        Retrieve a list of recent inspection activities.
        
        :param limit: Number of records to return
        :return: List of summarized inspection results
        """
        with self.getConnection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT i.inspection_id, i.image_path, i.inspection_date, COUNT(id.id) as defect_count,
                GROUP_CONCAT(DISTINCT d.defect_type) as types
                FROM inspection i
                LEFT JOIN inspection_defect id ON i.inspection_id = id.inspection_id
                LEFT JOIN defect d ON id.defect_id = d.defect_id
                GROUP BY i.inspection_id
                ORDER BY i.inspection_date DESC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                filename = os.path.basename(row[1]) if row[1] else "Unknown"
                results.append({
                    'id': row[0],
                    'filename': filename,
                    'time': row[2],
                    'count': row[3],
                    'types': row[4] or 'None',
                    'status': 'Flagged' if row[3] > 0 else 'Passed'
                })
            return results

class AsyncDatabaseBridge:
    """
    Asynchronous connector for SQL database (Architecture 3.2.2).
    """
    def __init__(self, dbPath="data/texvision.db"):
        self.db = TexVisionDB(dbPath)
        self.queue = queue.Queue()
        self.stopEvent = threading.Event()
        self.workerThread = threading.Thread(target=self._worker, daemon=True)
        self.workerThread.start()

    def _worker(self):
        """Internal background consumer task."""
        while not self.stopEvent.is_set() or not self.queue.empty():
            try:
                task = self.queue.get(timeout=1.0)
                if task['type'] == 'log_inspection':
                    self.db.logInspection(
                        task['filename'], 
                        task['filepath'], 
                        task['defects'], 
                        task['userId']
                    )
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Async DB Error: {e}")

    def logInspectionAsync(self, filename, filepath, defectsList, userId=1):
        """Enqueue data for non-blocking save."""
        self.queue.put({
            'type': 'log_inspection',
            'filename': filename,
            'filepath': filepath,
            'defects': defectsList,
            'userId': userId
        })

    def stop(self):
        """Terminate the async bridge gracefully."""
        self.stopEvent.set()
        self.workerThread.join(timeout=5.0)
