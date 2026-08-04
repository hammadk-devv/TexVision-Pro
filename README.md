<p align="center">
  <img src="assets/branding/hero-banner.png" width="100%" alt="TexVision-Pro Hero Banner">
</p>

<h1 align="center">TexVision-Pro</h1>

<p align="center">
<b>Industrial Computer Vision Platform for Automated Fabric Defect Detection</b>
</p>

<p align="center">
A Software Engineering Final Year Project that combines deep learning, real-time object detection, edge deployment, and a modern web dashboard to automate textile quality inspection.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)

![YOLOv8](https://img.shields.io/badge/YOLO-v8-success)

![Flask](https://img.shields.io/badge/Flask-Web_App-black)

![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-red)

![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

# Overview

TexVision-Pro is an end-to-end industrial computer vision platform designed to automate textile quality inspection using deep learning and real-time object detection.

The system combines **YOLOv8-based defect detection**, **Flask**, **SQLite**, and **Raspberry Pi 5 edge deployment** into a modular inspection platform capable of identifying defects with speed, consistency, and reliability.

Unlike a standalone machine learning model, TexVision-Pro was engineered as a complete software system, integrating AI, backend development, data management, visualization, and deployment into a production-oriented workflow.

---

# Key Features

- Real-time fabric defect detection using YOLOv8
- Support for YOLOv8m and YOLOv8n models
- Live camera-based inspection
- Raspberry Pi 5 edge deployment
- Flask-powered management dashboard
- Inspection history and analytics
- User management system
- Modular software architecture
- Configurable training pipeline
- Industrial-oriented workflow

---

# System Architecture

<p align="center">
<img src="assets/diagrams/system-architecture.png" width="100%">
</p>

---

# Inference Pipeline

<p align="center">
<img src="assets/diagrams/inference-pipeline.png" width="100%">
</p>

---

# Deployment Architecture

<p align="center">
<img src="assets/diagrams/deployment.png" width="100%">
</p>

---

# Model Training Pipeline

<p align="center">
<img src="assets/diagrams/training-pipeline.png" width="100%">
</p>

---

# Dashboard Preview

## Login

<p align="center">
<img src="assets/screenshots/authentication/login.png" width="70%">
</p>

---

## Dashboard

| Dashboard | Detection |
|------------|------------|
| <img src="assets/screenshots/dashboard/dashboard.png"> | <img src="assets/screenshots/dashboard/dashboard-detection.png"> |

---

| Analytics | History |
|------------|------------|
| <img src="assets/screenshots/dashboard/analytics.png"> | <img src="assets/screenshots/dashboard/history.png"> |

---

| User Management | Settings |
|------------|------------|
| <img src="assets/screenshots/dashboard/usermanagement.png"> | <img src="assets/screenshots/dashboard/usersettings.png"> |

---

## Live Detection

<p align="center">
<img src="assets/detection/livedetection.png" width="90%">
</p>

---

# Performance

| Model | Precision | Recall | Deployment |
|--------|----------:|-------:|------------|
| YOLOv8m | **94%** | **89%** | Raspberry Pi 5 / Desktop |
| YOLOv8n | **93%** | **88%** | Raspberry Pi 5 / Desktop |

---

# Technology Stack

| Category | Technologies |
|------------|-------------|
| Programming Language | Python |
| Computer Vision | YOLOv8 |
| Backend | Flask |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Edge Computing | Raspberry Pi 5 |
| Development Tools | Git, VS Code |

---

# Repository Structure

```text
TexVision-Pro
│
├── assets/
├── configs/
├── docs/
├── scripts/
├── src/
├── tests/
├── README.md
├── requirements.txt
└── LICENSE