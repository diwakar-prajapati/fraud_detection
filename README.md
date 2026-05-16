# 🛡️ Real-Time Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 Overview

A **production-ready, real-time fraud detection system** that identifies fraudulent credit card transactions using multiple Machine Learning and Deep Learning models. The system includes a REST API for real-time predictions and an interactive dashboard for live transaction monitoring.

### 🎯 Key Features

- **Multiple ML Models**: XGBoost, Random Forest, Logistic Regression, SVM, Neural Network, Rule-Based
- **Real-time API**: FastAPI endpoints with <50ms response time
- **Interactive Dashboard**: Streamlit dashboard with live monitoring
- **Live User Input**: Accept customer transactions in real-time
- **Model Comparison**: Compare 6 different models side by side
- **Statistical Analysis**: Hypothesis testing, probability distributions, correlation analysis
- **Advanced Visualizations**: 3D plots, heatmaps, time series, parallel coordinates
- **Imbalanced Data Handling**: SMOTE for class imbalance (0.17% fraud rate)

### 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | 99.96% | 0.99 | 0.82 | 0.89 |
| **Random Forest** | 99.94% | 0.98 | 0.80 | 0.88 |
| **Neural Network** | 99.94% | 0.98 | 0.81 | 0.88 |
| **Logistic Regression** | 97.23% | 0.85 | 0.72 | 0.78 |
| **SVM** | 96.89% | 0.83 | 0.70 | 0.76 |
| **Rule-Based** | 85.00% | 0.70 | 0.65 | 0.67 |

---

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────────────────┐
│ FRAUD DETECTION SYSTEM │
├─────────────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Mobile App │────▶│ │ │ │ │
│ │ Web Form │ │ FastAPI │────▶│ XGBoost │ │
│ │ API Call │────▶│ Server │ │ Random │ │
│ │ │ │ (Port 8000)│ │ Forest │ │
│ └──────────────┘ │ │ │ SVM │ │
│ └──────────────┘ │ Neural Net │ │
│ │ └──────────────┘ │
│ ▼ │ │
│ ┌──────────────┐ ▼ │
│ │ Streamlit │ ┌──────────────┐ │
│ │ Dashboard │ │ PostgreSQL │ │
│ │ (Port 8501) │ │ (Optional) │ │
│ └──────────────┘ └──────────────┘ │
│ │
└── 
───────────────────────────────────────────────────────────────────────┘


---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Installation

```bash
# 1. Clone the repository
git clone github.com/diwakar-prajapati/fraud_detection.git
cd fraud-detection-system

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download datasets
python scripts/download_datasets.py

# 6. Train models
python main.py

# 7. Run the system
python run_all.py