import csv
import os
from pathlib import Path
import json
from datetime import datetime
from io import StringIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

class ReportGenerator:
    """
    Automated reporting engine for TexVision-Pro.
    Refactored for Section 4.1 Coding Standards and Snippet 3 Reporting Format.
    """
    def __init__(self, db):
        """
        :param db: TexVisionDB instance
        """
        self.db = db
        # Use absolute path for reports directory to avoid ambiguity
        baseDir = Path(__file__).parent.parent.parent
        self.reportsDir = os.path.join(str(baseDir), "reports")
        os.makedirs(self.reportsDir, exist_ok=True)

    def generateReport(self, inspectionId: int) -> dict:
        """
        Section 4.3 / Snippet 3: Produce standardized JSON report.
        Useful for API integration and traceability.

        :param inspectionId: Target inspection record ID
        :return: JSON formatted dictionary as shown in documentation
        """
        with self.db.getConnection() as conn:
            cursor = conn.cursor()
            
            # Fetch base inspection details
            cursor.execute("""
                SELECT report_id, r.inspection_id, inspection_date, summary
                FROM report r
                JOIN inspection i ON r.inspection_id = i.inspection_id
                WHERE r.inspection_id = ?
            """, (inspectionId,))
            
            reportRow = cursor.fetchone()
            if not reportRow:
                return {} # E4-US-1: Handle missing reports
            
            # Fetch defect list for this inspection
            cursor.execute("""
                SELECT id.id, d.defect_type, id.location_on_fabric, id.confidence
                FROM inspection_defect id
                JOIN defect d ON id.defect_id = d.defect_id
                WHERE id.inspection_id = ?
            """, (inspectionId,))
            
            defectRows = cursor.fetchall()
            defectsList = []
            for d in defectRows:
                try:
                    location = json.loads(d[2])
                except:
                    location = [0, 0, 0, 0] # Fallback
                    
                defectsList.append({
                    "defectId": d[0],
                    "type": d[1],
                    "location": location,
                    "confidence": round(d[3], 4)
                })
            
            # Final output matches Snippet 3 exactly
            reportData = {
                "reportId": reportRow[0],
                "inspectionId": reportRow[1],
                "inspectionDate": str(reportRow[2]),
                "defectsDetectedCount": len(defectsList),
                "defectsList": defectsList
            }
            return reportData

    def generateCsv(self, dateStr: str = None) -> tuple:
        """
        Generate a well-formatted CSV report for a given date.

        :param dateStr: Target date (YYYY-MM-DD)
        :return: (File path, string content)
        """
        if dateStr is None:
            dateStr = datetime.now().strftime('%Y-%m-%d')
            
        fileName = f"report_{dateStr}.csv"
        filePath = os.path.join(self.reportsDir, fileName)
        
        with self.db.getConnection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id.id, i.image_path, i.inspection_date, d.defect_type, 
                       ROUND(id.confidence, 4) as confidence, 
                       id.location_on_fabric as location, 
                       d.severity_level
                FROM inspection i
                JOIN inspection_defect id ON i.inspection_id = id.inspection_id
                JOIN defect d ON id.defect_id = d.defect_id
                WHERE date(i.inspection_date) = ?
                ORDER BY id.id ASC
            """, (dateStr,))
            
            rows = cursor.fetchall()
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(['ID', 'Filename', 'Timestamp', 'Defect Type', 'Confidence', 'Location (BBox)', 'Severity'])
            
            for r in rows:
                imageName = os.path.basename(r[1]) if r[1] else "Unknown"
                writer.writerow([r[0], imageName, r[2], r[3], r[4], r[5], r[6]])
            
            with open(filePath, 'w', newline='') as f:
                f.write(output.getvalue())
                
            return filePath, output.getvalue()

    def generatePdf(self, dateStr: str = None) -> str:
        """
        Generate PDF report for a given date (Section 4.3).

        :param dateStr: Target date (YYYY-MM-DD)
        :return: Absolute file path
        """
        if dateStr is None:
            dateStr = datetime.now().strftime('%Y-%m-%d')
        
        fileName = f"report_{dateStr}.pdf"
        filePath = os.path.join(self.reportsDir, fileName)
        
        with self.db.getConnection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT d.defect_type, i.image_path, id.confidence, d.severity_level
                FROM inspection i
                JOIN inspection_defect id ON i.inspection_id = id.inspection_id
                JOIN defect d ON id.defect_id = d.defect_id
                WHERE date(i.inspection_date) = ?
            """, (dateStr,))
            rows = cursor.fetchall()
        
        doc = SimpleDocTemplate(filePath, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        elements.append(Paragraph(f"<b>TexVision-Pro Production Summary</b>", styles['Title']))
        elements.append(Paragraph(f"Reporting Date: {dateStr}", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        if rows:
            data = [['Type', 'Filename', 'Confidence', 'Severity']]
            for r in rows:
                data.append([r[0], os.path.basename(r[1]), f"{r[2]:.2f}", r[3]])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            elements.append(table)
        else:
            elements.append(Paragraph("No defects detected.", styles['Normal']))
        
        doc.build(elements)
        return filePath
