"""
TexVision-Pro Production Web Dashboard
Matches Figure 2.6 with Premium Gold/Brown Theme
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for, send_file
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_wtf import FlaskForm, CSRFProtect
from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask_wtf.csrf import generate_csrf
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, Length, EqualTo
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from pathlib import Path
import sys
import base64
from datetime import datetime

# Section 4.2: Configuration Management
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.integrated_detector import IntegratedDetector, ImageProcessor
from deployment.database import TexVisionDB
from deployment.reports import ReportGenerator

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_texvision_key')
CORS(app)
bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)

# --- Forms (Section 4.1 / Section 4.2) ---
class LoginForm(FlaskForm):
    username = StringField('Email Address', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    role = SelectField('Role', choices=[('Admin', 'Admin'), ('Operator', 'Operator')], default='Admin')
    submit = SubmitField('Login')

class ResetEmailForm(FlaskForm):
    email = StringField('Email Address', validators=[DataRequired(), Email()])
    submit = SubmitField('Send OTP')

class VerifyOTPForm(FlaskForm):
    otp = StringField('Enter OTP', validators=[DataRequired(), Length(min=6, max=6)])
    submit = SubmitField('Verify OTP')

class NewPasswordForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')

class DetectionForm(FlaskForm):
    image = FileField('Fabric Image', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')
    ])
    submit = SubmitField('Analyze')

# Configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Configuration from Environment (Section 4.2)
CONFIG = {
    'YOLO_PATH': os.getenv('YOLO_MODEL_PATH', 'models/yolov8s.pt'),
    'RESNET_PATH': os.getenv('RESNET_MODEL_PATH', 'models/checkpoints/best_model.pth'),
    'DEVICE': os.getenv('DEVICE', 'cpu'),
    'YOLO_CONF': float(os.getenv('YOLO_CONF', '0.50')), # Standardized to 0.50
    'RESNET_CONF': float(os.getenv('RESNET_CONF', '0.80'))
}

# Lazy-loaded singleton
_detectorInstance = None

def getDetector():
    """Retrieve or initialize the singleton detector instance (camelCase)"""
    global _detectorInstance
    if _detectorInstance is None:
        print(f"🚀 Initializing TexVision-Pro Engine (Section 4.1)...")
        _detectorInstance = IntegratedDetector(
            yoloModelPath=CONFIG['YOLO_PATH'],
            resnetModelPath=CONFIG['RESNET_PATH'],
            device=CONFIG['DEVICE'],
            yoloConf=CONFIG['YOLO_CONF'],
            resnetConf=CONFIG['RESNET_CONF']
        )
    return _detectorInstance

# Initialize DB and Reports early (Using Absolute Paths)
basePath = Path(__file__).parent.parent.parent
dbPath = os.path.join(str(basePath), os.getenv('DATABASE_PATH', 'data/texvision.db'))
db = TexVisionDB(dbPath=dbPath)
reportGenerator = ReportGenerator(db)

# --- Utilities (Section 3.6 / camelCase) ---

def validateImageFormat(filename):
    """Workflow 3.6.1: Check if format is JPG/PNG"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ['.jpg', '.jpeg', '.png']

def checkImageQuality(image, threshold=50):
    """Workflow 3.6.2: Detect blur using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > threshold, variance

# Pre-load detector to avoid lag on first upload
with app.app_context():
    try:
        getDetector()
    except Exception as e:
        print(f"⚠️ Warning: Detector background initialization failure: {e}")

# --- Routes (Section 4.3) ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data # Labeled 'Email Address' in UI
        password = form.password.data
        role = form.role.data
        
        # Maintain existing demo logic but allow any email containing 'admin' for convenience
        # or stick strictly to 'admin' if required. The user says "Email Address" field.
        # We'll allow 'admin@texvision.com' or just 'admin' to keep it fast for them.
        validUser = (username.lower() == 'admin@texvision.com' or username.lower() == 'admin')
        demoHash = bcrypt.generate_password_hash('admin123').decode('utf-8')
        
        if validUser and bcrypt.check_password_hash(demoHash, password):
            session['logged_in'] = True
            session['user'] = role
            session['role'] = role
            return redirect(url_for('dashboard'))
        return serve_login_page(form, error="Invalid email or password")
    
    return serve_login_page(form)

@app.route('/')
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    dailyStats = db.getDailyStats()
    recentInspections = db.getRecentInspections()
    return serve_dashboard_page(dailyStats, recentInspections)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (e.g., alert.wav)"""
    return send_from_directory('static', filename)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgotPassword():
    form = ResetEmailForm()
    if form.validate_on_submit():
        session['reset_email'] = form.email.data
        session['reset_otp'] = "123456" # Simulated OTP
        return redirect(url_for('verifyOtp'))
    return serve_reset_page(step=1, form=form)

@app.route('/verify-otp', methods=['GET', 'POST'])
def verifyOtp():
    if 'reset_email' not in session:
        return redirect(url_for('forgotPassword'))
    
    form = VerifyOTPForm()
    if form.validate_on_submit():
        if form.otp.data == session.get('reset_otp'):
            session['otp_verified'] = True
            return redirect(url_for('resetPassword'))
        return serve_reset_page(step=2, form=form, error="Invalid OTP code")
    return serve_reset_page(step=2, form=form)

@app.route('/reset-password', methods=['GET', 'POST'])
def resetPassword():
    if not session.get('otp_verified'):
        return redirect(url_for('forgotPassword'))
    
    form = NewPasswordForm()
    if form.validate_on_submit():
        # In a real app, update DB here. 
        # Requirement: "DO NOT alter existing business logic" 
        # So we just show success and redirect.
        return serve_reset_page(step=3, form=form, success=True)
    return serve_reset_page(step=3, form=form)

@app.route('/detect', methods=['POST']) # Section 4.3 / Snippet 1
def detect():
    """Handles file uploads, validates format, and processes image using CNN and YOLO"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized action'}), 401
        
    try:
        # Step 1: Check for image in request (Snippet 1 logic)
        if 'image' not in request.files:
            return jsonify({'error': 'No image found (Snippet 1)'}), 400
            
        fabricFile = request.files['image']
        if fabricFile and validateImageFormat(fabricFile.filename):
            filename = secure_filename(fabricFile.filename)
            filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            fabricFile.save(filePath)
            
            # Step 2: Analysis Workflow
            engine = getDetector()
            rawImage = cv2.imread(filePath)
            
            # ImageProcessor (Section 3.8 / Figure 3.3)
            processedImage = ImageProcessor.processImage(rawImage)
            
            # Quality Check (Workflow 3.6.2)
            qualityOk, qualityScore = checkImageQuality(processedImage)
            warningMsg = None
            if not qualityOk:
                warningMsg = f"Poor image quality (Score: {qualityScore:.1f}). Accurate detection may vary."
            
            # DefectDetectionModule (Section 3.8 / Figure 3.3)
            # Classification (CNN) followed by Localization (YOLO) happens inside
            analysisResults = engine.detectAndClassify(processedImage)
            
            # Visualize Results (Section 4.3 Visualization)
            resFilename = f"res_{filename}"
            resPath = os.path.join(app.config['RESULTS_FOLDER'], resFilename)
            annotatedImage = engine.visualize(processedImage, analysisResults, savePath=resPath)
            
            # Log To Database (Normalized Schema Section 3.5)
            db.logInspection(filename, filePath, analysisResults)
            
            # Encode for Dashboard Presentation
            _, buffer = cv2.imencode('.jpg', annotatedImage)
            imgBase64 = base64.b64encode(buffer).decode('utf-8')
            
            currentStats = db.getDailyStats()
            
            # Multi-part JSON Response (Section 4.3 / Snippet 1)
            return jsonify({
                'success': True,
                'image': f'data:image/jpeg;base64,{imgBase64}',
                'count': len(analysisResults),
                'stats': currentStats,
                'detections': analysisResults,
                'warning': warningMsg
            })
            
        return jsonify({'error': 'Invalid file type (JPG/PNG required)'}), 400
        
    except Exception as e:
        print(f"❌ Detect Error: {e}")
        return jsonify({'error': f'Processing failure: {str(e)}'}), 500

@app.route('/api/historical_data', methods=['GET'])
def getHistoricalData():
    """Fetch last 30 days of defect rate data for historical chart"""
    try:
        conn = db.getConnection()
        cursor = conn.cursor()
        
        # Get daily defect rates for last 30 days
        query = """
        SELECT 
            DATE(inspection_date) as date,
            COUNT(*) as total_inspections,
            SUM(CASE WHEN defect_count > 0 THEN 1 ELSE 0 END) as defective_count
        FROM (
            SELECT 
                i.inspection_date,
                COUNT(id.defect_id) as defect_count
            FROM inspection i
            LEFT JOIN inspection_defect id ON i.inspection_id = id.inspection_id
            WHERE i.inspection_date >= date('now', '-30 days')
            GROUP BY i.inspection_id, i.inspection_date
        )
        GROUP BY DATE(inspection_date)
        ORDER BY date ASC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Calculate defect rates
        data = []
        for row in rows:
            date_str = row[0]
            total = row[1]
            defective = row[2]
            defect_rate = (defective / total * 100) if total > 0 else 0
            data.append({
                'date': date_str,
                'defect_rate': round(defect_rate, 2),
                'total_inspections': total
            })
        
        # Calculate average
        avg_rate = sum(d['defect_rate'] for d in data) / len(data) if data else 0
        
        return jsonify({
            'success': True,
            'data': data,
            'average_rate': round(avg_rate, 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/export/csv')
def exportCsv():
    dateStr = request.args.get('date')
    filePath, content = reportGenerator.generateCsv(dateStr)
    filename = f"texvision_report_{dateStr if dateStr else datetime.now().strftime('%Y%m%d')}.csv"
    return content, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': f'attachment; filename={filename}'
    }

@app.route('/export/pdf')
def exportPdf():
    """E2-US-2: PDF report generation"""
    dateStr = request.args.get('date')
    path = reportGenerator.generatePdf(dateStr)
    fileName = f"texvision_report_{dateStr if dateStr else datetime.now().strftime('%Y%m%d')}.pdf"
    return send_file(path, as_attachment=True, download_name=fileName)

# --- Premium UI Pages (Section 2.8 / Section 4.3) ---

def serve_login_page(form, error=None):
    errorHtml = f'<div style="color: #721c24; background: #f8d7da; padding: 12px; border: 1px solid #f5c6cb; border-radius: 6px; margin-bottom: 20px; font-size: 14px; text-align: left;">{error}</div>' if error else ''
    csrfHtml = form.csrf_token() if form else "" 
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TexVision-Pro | Secure Access</title>
    <style>
        :root {{ 
            --primary: #a6914b; 
            --bg: #f4eee1; 
            --text: #4a371b; 
            --input-border: #ddd;
            --input-focus: #a6914b;
            --card-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }}
        body {{ 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            background: var(--bg); 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            min-height: 100vh; 
            margin: 0; 
            color: var(--text); 
        }}
        .login-card {{ 
            background: #ffffff; 
            width: 100%; 
            max-width: 400px; 
            border-radius: 8px; 
            border: 1px solid #e0d0b0;
            box-shadow: var(--card-shadow); 
            overflow: hidden; 
            margin: 20px;
        }}
        .header {{ 
            background: var(--primary); 
            color: white; 
            padding: 35px 20px; 
            text-align: center; 
        }}
        .header svg {{ width: 48px; height: 48px; margin-bottom: 12px; fill: white; }}
        .header h1 {{ margin: 0; font-size: 22px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; }}
        .header p {{ margin: 8px 0 0; font-size: 13px; opacity: 0.9; }}
        .content {{ padding: 30px; }}
        .banner {{ 
            background: #fbf9f2; 
            border-left: 4px solid var(--primary); 
            padding: 15px; 
            border-radius: 4px; 
            margin-bottom: 25px; 
            display: flex; 
            align-items: flex-start; 
            gap: 12px; 
            font-size: 13px; 
            border-top: 1px solid #eee;
            border-right: 1px solid #eee;
            border-bottom: 1px solid #eee;
        }}
        .banner-icon {{ color: var(--primary); font-size: 16px; font-weight: bold; padding-top: 2px; }}
        .banner-text b {{ display: block; margin-bottom: 4px; font-size: 14px; color: #4a371b; }}
        .form-group {{ margin-bottom: 18px; text-align: left; }}
        .form-group label {{ display: block; font-size: 14px; margin-bottom: 8px; font-weight: 600; color: #5d4f37; }}
        input, select {{ 
            width: 100%; 
            height: 40px; 
            padding: 0 12px; 
            border: 1px solid var(--input-border); 
            border-radius: 6px; 
            box-sizing: border-box; 
            font-size: 14px; 
            color: #333; 
            background-color: #fdfdfd;
            transition: border-color 0.2s, box-shadow 0.2s;
            font-family: inherit;
        }}
        input:focus, select:focus {{ 
            outline: none; 
            border-color: var(--input-focus); 
            box-shadow: 0 0 0 3px rgba(166, 145, 75, 0.15); 
            background-color: #fff;
        }}
        .forgot-link {{ text-align: right; margin-top: -10px; margin-bottom: 22px; }}
        .forgot-link a {{ font-size: 12px; color: var(--primary); text-decoration: none; font-weight: 700; transition: color 0.2s; }}
        .forgot-link a:hover {{ color: #8a783d; text-decoration: underline; }}
        button {{ 
            width: 100%; 
            padding: 12px; 
            background: var(--primary); 
            color: white; 
            border: none; 
            border-radius: 6px; 
            cursor: pointer; 
            font-size: 15px; 
            font-weight: 700; 
            margin-top: 5px; 
            transition: background 0.2s, transform 0.1s; 
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        button:hover {{ background: #8e7b40; }}
        button:active {{ transform: translateY(1px); }}
        .status-bar {{ 
            background: #f7f4ec; 
            padding: 12px; 
            border-radius: 6px; 
            border: 1px solid #e8e2d4;
            text-align: center; 
            margin-top: 25px; 
            font-size: 13px; 
            font-weight: 600; 
            color: #7a6e55; 
        }}
    </style>
</head>
<body>
    <div class="login-card">
        <div class="header">
            <svg viewBox="0 0 24 24"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 2.18l7 3.12v4.7c0 4.67-3.13 8.75-7 9.81-3.87-1.06-7-5.14-7-9.81V6.3l7-3.12z"/></svg>
            <h1>TEXVISION PRO</h1>
            <p>Industrial Quality Control Platform</p>
        </div>
        <div class="content">
            <div class="banner">
                <div class="banner-icon">ⓘ</div>
                <div class="banner-text">
                    <b>Authorized Access Only</b>
                    All login attempts are logged and monitored. Please enter your credentials to continue.
                </div>
            </div>
            {errorHtml}
            <form method="POST" action="/login">
                {csrfHtml}
                <div class="form-group">
                    <label for="username">Email Address</label>
                    <input type="email" id="username" name="username" placeholder="name@company.com" required>
                </div>
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" placeholder="••••••••" required>
                </div>
                <div class="forgot-link">
                    <a href="/forgot-password">Forgot Password?</a>
                </div>
                <div class="form-group">
                    <label for="role">Control Role</label>
                    <select id="role" name="role">
                        <option value="Admin">System Administrator</option>
                        <option value="Operator">Line Operator</option>
                    </select>
                </div>
                <button type="submit">Log Into System</button>
            </form>
            <div class="status-bar">
                System Status: Ready
            </div>
        </div>
    </div>
</body>
</html>
"""

def serve_reset_page(step, form=None, error=None, success=False):
    csrfHtml = form.csrf_token() if form else ""
    errorHtml = f'<div style="color: #721c24; background: #f8d7da; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-size: 14px; border: 1px solid #f5c6cb;">{error}</div>' if error else ''
    successHtml = f'<div style="background: #d4edda; color: #155724; padding: 12px; border-radius: 8px; margin-top: 15px; font-size: 14px; text-align: center; border: 1px solid #c3e6cb;">Password reset successful! Redirecting to login... <script>setTimeout(() => {{ window.location.href = "/login"; }}, 3000);</script></div>' if success else ''
    
    email = session.get('reset_email', 'user@example.com')
    
    content = ""
    if step == 1:
        content = f"""
            <div class="reset-info">Enter your email to receive OTP</div>
            <a href="/login" class="back-link">← Back to Login</a>
            <form method="POST" action="/forgot-password">
                {csrfHtml}
                <div class="form-group">
                    <label>Email Address</label>
                    <div style="position: relative;">
                        <span style="position: absolute; left: 12px; top: 12px; color: #999;">✉</span>
                        <input type="email" name="email" placeholder="Enter your registered email" style="padding-left: 35px;" required>
                    </div>
                </div>
                <p style="font-size: 12px; opacity: 0.7; margin-bottom: 20px;">We'll send a 6-digit OTP to this email address</p>
                <button type="submit">Send OTP</button>
            </form>
        """
    elif step == 2:
        content = f"""
            <div class="reset-info">Enter the OTP sent to your email</div>
            <a href="/forgot-password" class="back-link">← Back to Login</a>
            <div class="banner">
                OTP has been sent to: <b>{email}</b>
            </div>
            {errorHtml}
            <form method="POST" action="/verify-otp">
                {csrfHtml}
                <div class="form-group">
                    <label>Enter OTP</label>
                    <input type="text" name="otp" placeholder="000000" style="text-align: center; letter-spacing: 15px; font-size: 24px;" maxlength="6" required>
                </div>
                <p style="font-size: 12px; opacity: 0.7; margin-bottom: 20px; text-align: center;">Enter the 6-digit code sent to your email</p>
                <button type="submit">Verify OTP</button>
                <a href="#" style="display: block; text-align: center; color: var(--primary); text-decoration: none; font-size: 13px; margin-top: 15px; font-weight: bold;">Resend OTP</a>
            </form>
        """
    elif step == 3:
        content = f"""
            <div class="reset-info">Create your new password</div>
            <a href="/verify-otp" class="back-link">← Back to Login</a>
            <div class="banner">
                <div class="banner-icon">🛡</div>
                <div class="banner-text">Create a strong password with at least 8 characters</div>
            </div>
            {errorHtml}
            <form method="POST" action="/reset-password">
                {csrfHtml}
                <div class="form-group">
                    <label>New Password</label>
                    <input type="password" name="password" placeholder="Enter new password" required>
                </div>
                <div class="form-group">
                    <label>Confirm Password</label>
                    <input type="password" name="confirm_password" placeholder="Confirm new password" required>
                </div>
                <button type="submit">Reset Password</button>
            </form>
            {successHtml}
        """

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>TexVision-Pro | Reset Password</title>
    <style>
        :root {{ --primary: #a6914b; --bg: #f4eee1; --text: #5d4f37; --card: #ffffff; --banner: #ece6d4; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: var(--bg); display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; color: var(--text); flex-direction: column; }}
        .reset-card {{ background: #fdfaf3; width: 400px; border-radius: 20px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.05); padding-bottom: 20px; }}
        .header {{ background: var(--primary); color: white; padding: 30px 20px; text-align: center; }}
        .header svg {{ width: 50px; height: 50px; margin-bottom: 10px; fill: white; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header p {{ margin: 5px 0 0; font-size: 14px; opacity: 0.9; }}
        .content {{ padding: 30px; }}
        .reset-info {{ text-align: center; font-size: 14px; margin-bottom: 20px; display: none; }}
        .back-link {{ display: block; margin-bottom: 20px; color: var(--primary); text-decoration: none; font-size: 13px; font-weight: 500; }}
        .banner {{ background: var(--banner); border-left: 5px solid var(--primary); padding: 15px; border-radius: 4px; margin-bottom: 20px; display: flex; align-items: center; gap: 12px; font-size: 14px; }}
        .banner-icon {{ color: var(--primary); font-size: 18px; }}
        .form-group {{ margin-bottom: 20px; }}
        .form-group label {{ display: block; font-size: 14px; margin-bottom: 8px; font-weight: 500; }}
        input {{ width: 100%; padding: 12px; border: 1px solid #d8cfbc; border-radius: 8px; box-sizing: border-box; font-size: 14px; color: var(--text); font-family: inherit; }}
        button {{ width: 100%; padding: 14px; background: var(--primary); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold; margin-top: 10px; }}
        .footer {{ margin-top: 20px; font-size: 12px; opacity: 0.5; text-align: center; }}
    </style>
</head>
<body>
    <div class="reset-card">
        <div class="header">
            <svg viewBox="0 0 24 24"><path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zM9 6c0-1.66 1.34-3 3-3s3 1.34 3 3v2H9V6zm9 14H6V10h12v10zm-6-3c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2z"/></svg>
            <h1>Reset Password</h1>
            <p>{ 'Enter your email to receive OTP' if step == 1 else 'Enter the OTP sent to your email' if step == 2 else 'Create your new password' }</p>
        </div>
        <div class="content">
            {content}
        </div>
    </div>
    <div class="footer">
        System Version 2.1.0 | © 2025 Industrial Automation Systems
    </div>
</body>
</html>
"""

def serve_dashboard_page(stats, recent):
    userRole = session.get('role', 'Operator')
    csrfToken = generate_csrf()
    
    rows = ""
    for r in recent:
        rows += f"<tr><td>{r['id']}</td><td>{r['filename']}</td><td>{r['count']}</td><td>{r['types']}</td><td><span class='badge {r['status']}'>{r['status']}</span></td></tr>"

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>TexVision-Pro Dashboard</title>
    <style>
        :root {{ --primary: #6b5226; --bg: #f4eee0; --accent: #c9b07a; --text: #4a371c; }}
        body {{ font-family: 'Segoe UI'; background: var(--bg); margin: 0; color: var(--text); }}
        header {{ background: var(--primary); color: white; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }}
        .nav-info {{ display: flex; gap: 20px; font-size: 14px; opacity: 0.8; padding: 10px 30px; background: #e6dfcc; border-bottom: 1px solid var(--accent); }}
        main {{ padding: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); border-top: 4px solid var(--primary); }}
        .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px; }}
        .stat-box {{ background: white; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid var(--accent); }}
        .stat-val {{ font-size: 24px; font-weight: bold; color: var(--primary); }}
        .stat-lbl {{ font-size: 12px; opacity: 0.7; }}
        .upload-zone {{ border: 2px dashed var(--accent); border-radius: 8px; padding: 40px; text-align: center; cursor: pointer; background: #faf9f5; }}
        .upload-zone:hover {{ background: #f4f2ea; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
        th {{ text-align: left; background: #f4f2ea; padding: 10px; border-bottom: 2px solid var(--accent); }}
        td {{ padding: 10px; border-bottom: 1px solid #eee; }}
        .badge {{ padding: 3px 8px; border-radius: 12px; font-size: 11px; }}
        .Passed {{ background: #d4edda; color: #155724; }}
        .Flagged {{ background: #f8d7da; color: #721c24; }}
        .slider {{ width: 100%; margin: 15px 0; accent-color: var(--primary); }}
        #resImg {{ width: 100%; border-radius: 5px; margin-top: 15px; display: none; }}
        .btn-export {{ background: var(--primary); color: white; padding: 8px 15px; text-decoration: none; border-radius: 4px; font-size: 12px; }}
        @keyframes flash-red {{
            0%, 100% {{ background-color: var(--bg); }}
            50% {{ background-color: #ff000033; }}
        }}
        .alert-flash {{
            animation: flash-red 0.5s ease-in-out 3;
        }}
        .alert-flash {{
            animation: flash-red 0.5s ease-in-out 3;
        }}
    </style>
</head>
<body>
    <audio id="alertSound" preload="auto">
        <source src="/static/alert.wav" type="audio/wav">
    </audio>
    
    <header>
        <div style="font-size: 20px; font-weight: bold;">🛡️ TEXVISION PRO | Control Dashboard</div>
        <div style="font-size: 14px;">User: <b>Admin</b> | <a href="/logout" style="color:white; text-decoration:none;">Logout</a></div>
    </header>
    <div class="nav-info">
        <span>✅ System: Online</span>
        <span>✅ Camera: Connected</span>
        <span>✅ Processing: Ready</span>
        <span style="margin-left:auto">Uptime: 47h 23m</span>
    </div>

    <div id="alertAck" style="display:none; position:fixed; top:20px; left:50%; transform:translateX(-50%); background:red; color:white; padding:15px; border-radius:5px; cursor:pointer; z-index:999; box-shadow:0 5px 15px rgba(0,0,0,0.3)" onclick="stopAlert()">
        ⚠️ DEFECT WARNING! Click to Acknowledge
    </div>

    <main>
        <div class="left-col">
            <div class="card">
                <h3>📤 Analyze Fabric Image</h3>
                <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInp').click()">
                    <div style="font-size: 40px;">☁️</div>
                    <p>Click to Upload Fabric Image<br><small>Supports JPG, PNG (Max 50MB)</small></p>
                    <input type="file" id="fileInp" style="display:none" onchange="handleUpload(this)">
                </div>
                <div id="statusMsg" style="margin-top:10px; font-size:12px; color:var(--primary)">Ready to analyze</div>
                <img id="resImg">
            </div>

            <div class="card" style="margin-top:20px">
                <div style="display:flex; justify-content:space-between; align-items:center">
                    <h3>📊 Production Reports</h3>
                    <div style="display:flex; gap:5px; align-items:center">
                        <input type="date" id="reportDate" style="padding:5px; border:1px solid #ccc; border-radius:4px">
                        <a href="#" onclick="exportReport('csv')" class="btn-export">CSV</a>
                        <a href="#" onclick="exportReport('pdf')" class="btn-export" style="background:#8b6914">PDF</a>
                    </div>
                </div>
                <table>
                    <thead>
                        <tr><th>ID</th><th>Filename</th><th>Defects</th><th>Types</th><th>Status</th></tr>
                    </thead>
                    <tbody id="insTable">
                        {rows}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="right-col">
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-val" id="stat-total">{stats['total_inspections']}</div>
                    <div class="stat-lbl">Total Samples</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val" id="stat-found" style="color:#721c24">{stats['defects_found']}</div>
                    <div class="stat-lbl">Defects Found</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val" id="stat-rate">{stats['pass_rate']}%</div>
                    <div class="stat-lbl">Efficiency Rate</div>
                </div>
            </div>


            <!-- Confidence Score Display Section -->
            <div class="card" id="confidenceCard" style="display:none; margin-top:20px">
                <h3>🎯 Detection Confidence</h3>
                <div id="confidenceList" style="margin-top:15px">
                    <!-- Dynamically populated with confidence scores -->
                </div>
            </div>

            <!-- Historical Comparison Chart Section -->
            <div class="card" style="margin-top:20px">
                <h3>📊 30-Day Defect Trend</h3>
                <canvas id="historicalChart" width="400" height="200" style="margin-top:15px"></canvas>
                <div id="trendIndicator" style="margin-top:10px; padding:10px; background:#f9f9f9; border-radius:4px; text-align:center; font-size:13px">
                    Loading historical data...
                </div>
            </div>


            <div class="card" style="margin-top:20px; min-height:200px">
                <h3>📝 Detection Log</h3>
                <div id="log" style="background:#4a371c; color:#00ff00; padding:15px; font-family:monospace; font-size:12px; border-radius:5px; height:150px; overflow-y:auto">
                    > Secure System initialized.
                </div>
            </div>
        </div>
    </main>

    <input type="hidden" id="csrf_token" value="{csrfToken}">

    <script>
        function log(msg) {{
            const div = document.getElementById('log');
            div.innerHTML += `<div>> ${{msg}}</div>`;
            div.scrollTop = div.scrollHeight;
        }}

        function exportReport(type) {{
            const date = document.getElementById('reportDate').value;
            let url = `/export/${{type}}`;
            if (date) url += `?date=${{date}}`;
            window.location.href = url;
        }}

        function stopAlert() {{
            document.body.classList.remove('alert-flash');
            document.getElementById('alertSound').pause();
            document.getElementById('alertSound').currentTime = 0;
            document.getElementById('alertAck').style.display = 'none';
        }}


        function handleUpload(inp) {{
            if (!inp.files[0]) return;
            const formData = new FormData();
            formData.append('image', inp.files[0]);
            
            log(`Analyzing ${{inp.files[0].name}}...`);
            document.getElementById('statusMsg').innerText = "Running CNN + YOLO...";

            const csrfToken = document.getElementById('csrf_token').value;
            fetch('/detect', {{ 
                method: 'POST', 
                body: formData,
                headers:{{
                    'X-CSRFToken': csrfToken
                }}
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    document.getElementById('resImg').src = data.image;
                    document.getElementById('resImg').style.display = "block";
                    document.getElementById('stat-total').innerText = data.stats.total_inspections;
                    document.getElementById('stat-found').innerText = data.stats.defects_found;
                    document.getElementById('stat-rate').innerText = data.stats.pass_rate + "%";
                    
                    log(`Analysis success. Count: ${{data.count}}`);
                    if (data.warning) log(`⚠️ ${{data.warning}}`);
                    
                    // Update confidence display with detection data
                    if (data.detections && data.detections.length > 0) {{
                        updateConfidenceDisplay(data.detections);
                    }}
                    
                    if (data.count > 0) {{
                        document.body.classList.add('alert-flash');
                        document.getElementById('alertSound').play().catch(e => {{}});
                        document.getElementById('alertAck').style.display = 'block';
                    }}
                    document.getElementById('statusMsg').innerText = "Ready to analyze";
                }} else {{
                    alert("Error: " + data.error);
                }}
            }});
        }}

        // Update confidence display with detection results
        function updateConfidenceDisplay(detections) {{
            const confidenceCard = document.getElementById('confidenceCard');
            const confidenceList = document.getElementById('confidenceList');
            
            if (!detections || detections.length === 0) {{
                confidenceCard.style.display = 'none';
                return;
            }}
            
            confidenceCard.style.display = 'block';
            confidenceList.innerHTML = '';
            
            detections.forEach(det => {{
                const confidence = Math.round(det.confidence * 100);
                let barColor = '#dc3545'; // Red for <70%
                if (confidence > 90) barColor = '#28a745'; // Green for >90%
                else if (confidence >= 70) barColor = '#ffc107'; // Yellow for 70-90%
                
                const itemHtml = `
                    <div style="margin-bottom:12px; padding:10px; background:#f9f9f9; border-radius:4px">
                        <div style="display:flex; justify-content:space-between; margin-bottom:5px">
                            <span style="font-weight:600; font-size:13px">${{det.type || 'Defect'}}</span>
                            <span style="font-weight:700; color:${{barColor}}; font-size:13px">${{confidence}}%</span>
                        </div>
                        <div style="background:#e0e0e0; height:8px; border-radius:4px; overflow:hidden">
                            <div style="background:${{barColor}}; height:100%; width:${{confidence}}%; transition:width 0.3s"></div>
                        </div>
                    </div>
                `;
                confidenceList.innerHTML += itemHtml;
            }});
        }}

        // Load and render historical chart
        function loadHistoricalChart() {{
            fetch('/api/historical_data')
                .then(r => r.json())
                .then(data => {{
                    if (!data.success || !data.data || data.data.length === 0) {{
                        document.getElementById('trendIndicator').innerHTML = 'No historical data available';
                        return;
                    }}
                    
                    renderHistoricalChart(data.data, data.average_rate);
                    
                    // Update trend indicator
                    const latestRate = data.data[data.data.length - 1].defect_rate;
                    const avgRate = data.average_rate;
                    const diff = latestRate - avgRate;
                    const indicator = diff > 0 ? '📈 Worse' : '📉 Better';
                    const color = diff > 0 ? '#dc3545' : '#28a745';
                    
                    document.getElementById('trendIndicator').innerHTML = `
                        <span style="color:${{color}}; font-weight:600">${{indicator}} than average</span>
                        <span style="margin-left:10px; opacity:0.7">
                            Current: ${{latestRate.toFixed(1)}}% | Avg: ${{avgRate.toFixed(1)}}%
                        </span>
                    `;
                }})
                .catch(err => {{
                    console.error('Failed to load historical data:', err);
                    document.getElementById('trendIndicator').innerHTML = 'Failed to load data';
                }});
        }}

        // Render chart on canvas
        function renderHistoricalChart(data, avgRate) {{
            const canvas = document.getElementById('historicalChart');
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;
            
            // Clear canvas
            ctx.clearRect(0, 0, width, height);
            
            // Chart dimensions
            const padding = {{ top: 20, right: 20, bottom: 30, left: 40 }};
            const chartWidth = width - padding.left - padding.right;
            const chartHeight = height - padding.top - padding.bottom;
            
            // Find max rate for scaling
            const maxRate = Math.max(...data.map(d => d.defect_rate), avgRate) * 1.2;
            
            // Draw axes
            ctx.strokeStyle = '#ccc';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(padding.left, padding.top);
            ctx.lineTo(padding.left, height - padding.bottom);
            ctx.lineTo(width - padding.right, height - padding.bottom);
            ctx.stroke();
            
            // Draw average line (dashed)
            const avgY = height - padding.bottom - (avgRate / maxRate) * chartHeight;
            ctx.strokeStyle = '#6c757d';
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(padding.left, avgY);
            ctx.lineTo(width - padding.right, avgY);
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Draw data line
            ctx.strokeStyle = '#6b5226';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            data.forEach((point, i) => {{
                const x = padding.left + (i / (data.length - 1)) * chartWidth;
                const y = height - padding.bottom - (point.defect_rate / maxRate) * chartHeight;
                
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }});
            ctx.stroke();
            
            // Draw data points
            data.forEach((point, i) => {{
                const x = padding.left + (i / (data.length - 1)) * chartWidth;
                const y = height - padding.bottom - (point.defect_rate / maxRate) * chartHeight;
                
                ctx.fillStyle = (i === data.length - 1) ? '#c9b07a' : '#6b5226';
                ctx.beginPath();
                ctx.arc(x, y, (i === data.length - 1) ? 5 : 3, 0, 2 * Math.PI);
                ctx.fill();
            }});
            
            // Draw labels
            ctx.fillStyle = '#666';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            
            // X-axis labels (every 5 days)
            for (let i = 0; i < data.length; i += 5) {{
                const x = padding.left + (i / (data.length - 1)) * chartWidth;
                const date = new Date(data[i].date);
                const label = `${{date.getMonth() + 1}}/${{date.getDate()}}`;
                ctx.fillText(label, x, height - 10);
            }}
            
            // Y-axis label
            ctx.textAlign = 'right';
            ctx.fillText(`${{maxRate.toFixed(0)}}%`, padding.left - 5, padding.top + 10);
            ctx.fillText('0%', padding.left - 5, height - padding.bottom);
        }}

        // Load historical chart on page load
        window.addEventListener('DOMContentLoaded', loadHistoricalChart);
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
