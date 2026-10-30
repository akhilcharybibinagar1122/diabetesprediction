# 🩺 Diabetes Prediction System using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.x%20%7C%206.0-green.svg)](https://www.djangoproject.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](#)

An end-to-end clinical decision-support web application that predicts diabetes risk using Machine Learning algorithms (**Logistic Regression** and **Support Vector Machines**) trained on the **Pima Indians Diabetes Dataset**, delivered through a responsive **Django** web interface.

---

## 📑 Table of Contents

- [📌 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🔄 Operational Workflow](#-operational-workflow)
- [📊 Dataset & Biomarkers](#-dataset--biomarkers)
- [🔬 Machine Learning & Empirical Results](#-machine-learning--empirical-results)
- [📈 Visualizations](#-visualizations)
- [📁 Repository Structure](#-repository-structure)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
  - [Running Migrations](#running-migrations)
  - [Training / Verifying the Model](#training--verifying-the-model)
  - [Starting the Development Server](#starting-the-development-server)
- [🧪 Testing & Validation](#-testing--validation)
- [⚠️ Medical & Educational Disclaimer](#️-medical--educational-disclaimer)

---

## 📌 Project Overview

Traditional machine learning research often confines predictive models to offline Jupyter notebooks without evaluating real-world clinical usability. This project bridges the gap between academic classification research and production web deployment:

1. **Robust Clinical Sanitization:** Protects against unrealistic or invalid user inputs (e.g., negative values, physiological outliers, or missing attributes).
2. **Standardized Feature Pipeline:** Implements Scikit-Learn `Pipeline` wrapping `StandardScaler` to prevent feature scale distortion and data leakage.
3. **High-Performance Web Engine:** Implements an in-memory cached model singleton in Django to serve real-time predictions in sub-millisecond latency.
4. **Transparent Risk Communication:** Features state persistence (re-populating submitted forms) and clear medical advisory disclaimers.

---

## ✨ Key Features

- **8 Clinical Diagnostic Parameters:** Ingests pregnancies, glucose, blood pressure, skin fold thickness, insulin, BMI, pedigree function, and age.
- **Dual Algorithm Benchmarking:** Evaluates and compares **Logistic Regression** (linear interpretability) and **Support Vector Classifier** (RBF & Linear non-linear margin separation).
- **Automated Fallback Training:** If serialized model artifacts (`diabetes_model.joblib`) are absent, the application automatically invokes `train_model.py` to train, evaluate, and serialize the pipeline on startup.
- **Interactive Web Interface:** Clean HTML5, CSS3, and JavaScript templates styled with responsive design.
- **Comprehensive Academic Report:** Includes complete literature review, mathematical formulas, Data Flow Diagrams (DFD Level 0 & Level 1), and comparative benchmark matrices in [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

---

## 🏗️ System Architecture

```
+─────────────────────────────────────────────────────────────────────────────+
|                          SYSTEM ARCHITECTURE DIAGRAM                        |
+─────────────────────────────────────────────────────────────────────────────+

  [ Patient / Clinician ]
             │
             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │               Presentation Layer (Frontend)                 │
  │   - home.html: Landing page & system introduction           │
  │   - predict.html: 8-parameter clinical assessment form      │
  │   - Responsive UI feedback (Errors & Risk Badges)           │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ HTTP GET / POST
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 Django Application Backend                  │
  │   - urls.py: Clean URL routing (/ and /predict/result)      │
  │   - forms.py: Strict input sanitization & boundary checks   │
  │   - views.py: Singleton model cache & inference controller  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Validated Feature Vector
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │               Machine Learning Pipeline (.joblib)           │
  │   ┌─────────────────────────────────────────────────────┐   │
  │   │ Step 1: StandardScaler                              │   │
  │   │         Transforms raw features to z-scores         │   │
  │   └──────────────────────────┬──────────────────────────┘   │
  │                              ▼                              │
  │   ┌─────────────────────────────────────────────────────┐   │
  │   │ Step 2: Trained Classifier                          │   │
  │   │         - Logistic Regression (Linear Log-Odds)     │   │
  │   │         - Support Vector Machine (RBF Kernel)       │   │
  │   └─────────────────────────────────────────────────────┘   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Output: Binary Classification (0 / 1)
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                     Diagnostic Response                     │
  │   - Result Label: "Positive" (At Risk) or "Negative"        │
  │   - Form state preserved for ease of re-evaluation          │
  │   - Mandatory Medical Disclaimer rendered                   │
  └─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Operational Workflow

```
[ Raw Dataset ] ──> [ Preprocessing ] ──> [ Stratified Split ]
(diabetes.csv)      (StandardScaler)      (80% Train / 20% Test)
                                                    │
                     ┌──────────────────────────────┴─────────────────────────┐
                     ▼                                                         ▼
         [ Logistic Regression ]                                       [ Support Vector Machine ]
         - L2 Regularization                                           - RBF Kernel (gamma='scale')
         - Max Iterations = 1000                                       - Cost Parameter C = 1.0
                     │                                                         │
                     └──────────────────────────────┬──────────────────────────┘
                                                    │
                                                    ▼
                                     [ Performance Benchmarking ]
                                     - Accuracy, Precision, Recall, F1
                                     - ROC-AUC & Confusion Matrix
                                                    │
                                                    ▼
                                     [ Serialization & Export ]
                                     - diabetes_model.joblib
                                                    │
                                                    ▼
                                     [ Django Web Serving ]
                                     - Input sanitization (forms.py)
                                     - In-memory inference (views.py)
```

---

## 📊 Dataset & Biomarkers

The predictive model is trained on the **Pima Indians Diabetes Database** (compiled by the National Institute of Diabetes and Digestive and Kidney Diseases - NIDDK).

| Feature Name | Field | Measurement Unit | Clinical Description | Valid Range |
| :--- | :--- | :--- | :--- | :--- |
| **Pregnancies** | `n1` | Count | Number of times pregnant | 0 – 17 |
| **Glucose** | `n2` | mg/dL | 2-Hour post-challenge oral glucose tolerance | 40 – 250 |
| **Blood Pressure** | `n3` | mm Hg | Diastolic blood pressure | 40 – 140 |
| **Skin Thickness** | `n4` | mm | Triceps skin fold thickness | 0 – 99 |
| **Insulin** | `n5` | $\mu\text{U/mL}$ | 2-Hour serum insulin level | 0 – 850 |
| **BMI** | `n6` | $\text{kg/m}^2$ | Body Mass Index: $\text{Weight (kg)} / (\text{Height (m)})^2$ | 10.0 – 70.0 |
| **Pedigree Function** | `n7` | Score | Genetic diabetes risk score based on family history | 0.05 – 2.50 |
| **Age** | `n8` | Years | Patient age | 21 – 100 |

*Dataset Composition:* 768 patient records (500 non-diabetic [65.1%], 268 diabetic [34.9%]).

---

## 🔬 Machine Learning & Empirical Results

### Comparative Performance Matrix (Test Set: $N = 154$, Stratified 80:20 Split)

| Algorithm | Configuration | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF)** | $\gamma=\text{scale}, C=1.0$ | **75.32%** | **66.00%** | **61.11%** | **63.46%** | 0.7924 |
| **Support Vector Machine (Linear)** | Linear Kernel, $C=1.0$ | 72.08% | 62.22% | 51.85% | 56.57% | 0.8172 |
| **Logistic Regression** | $L_2$ Penalty, $C=1.0$ | 71.43% | 60.87% | 51.85% | 56.00% | **0.8230** |

### Confusion Matrix Breakdown ($N = 154$)

```
1. Logistic Regression Model:
                       Predicted Non-Diabetic    Predicted Diabetic
    Actual Non-Diabetic         82 (TN)                 18 (FP)
    Actual Diabetic             26 (FN)                 28 (TP)

2. Support Vector Machine (RBF Kernel):
                       Predicted Non-Diabetic    Predicted Diabetic
    Actual Non-Diabetic         83 (TN)                 17 (FP)
    Actual Diabetic             21 (FN)                 33 (TP)
```

### Feature Importance (Standardized Logistic Regression Coefficients)

$$\ln\left(\frac{p}{1-p}\right) = 1.144(\text{Glucose}) + 0.714(\text{BMI}) + 0.373(\text{Pregnancies}) + 0.256(\text{Pedigree}) + 0.184(\text{Age}) - \dots$$

1. **Glucose ($\beta = +1.144$):** Strongest positive driver of diabetes risk.
2. **BMI ($\beta = +0.714$):** Second strongest risk indicator.
3. **Pregnancies ($\beta = +0.373$):** Correlates with cumulative gestational glucose stress.
4. **Pedigree Function ($\beta = +0.256$):** Highlights significant hereditary genetic predisposition.

---

## 📈 Visualizations

The [`graphs/`](graphs/) directory contains high-resolution plots illustrating exploratory findings and model evaluation:

- **`graphs/eda_feature_distribution.png`**: Kernel density estimation (KDE) distributions for clinical attributes categorized by outcome.
- **`graphs/correlation_heatmap.png`**: Pearson correlation matrix identifying risk factor associations.
- **`graphs/model_performance_comparison.png`**: Bar chart comparison of Accuracy, Precision, Recall, and F1-score across models.
- **`graphs/roc_curves_comparison.png`**: Receiver Operating Characteristic curves comparing trade-offs between true positive and false positive rates.
- **`graphs/confusion_matrices_comparison.png`**: Side-by-side confusion matrix heatmaps showing true negatives, false positives, false negatives, and true positives.

---

## 📁 Repository Structure

```text
diabetesprediction/
├── .gitignore                                # Git ignore rules (venv, db.sqlite3, pycache)
├── README.md                                 # Comprehensive project documentation
├── PROJECT_REPORT.md                         # Full academic research report & DFD diagrams
├── requirements.txt                          # Project dependency specifications
├── main.py                                   # Helper script
├── powershell.cmd                            # Windows launcher script
│
├── graphs/                                   # High-resolution benchmark figures
│   ├── confusion_matrices_comparison.png
│   ├── correlation_heatmap.png
│   ├── eda_feature_distribution.png
│   ├── model_performance_comparison.png
│   └── roc_curves_comparison.png
│
└── diabetesprediction/                       # Django project root
    ├── manage.py                             # Django management CLI
    ├── diabetes.csv                          # Pima Indians Diabetes Dataset
    ├── diabetes_model.joblib                 # Serialized production ML pipeline
    ├── train_model.py                        # Standalone ML training & validation script
    ├── verify_workflow.py                    # Automated end-to-end integration test suite
    ├── requirements.txt                      # Application dependencies
    │
    ├── diabetesprediction/                   # Django core application package
    │   ├── __init__.py
    │   ├── asgi.py
    │   ├── forms.py                          # Clinical input validation & bounds checking
    │   ├── settings.py                       # Django configuration settings
    │   ├── tests.py                          # Django test cases
    │   ├── urls.py                           # URL routing configurations
    │   ├── views.py                          # Request handling & cached ML inference
    │   └── wsgi.py
    │
    ├── static/                               # Static images and stylesheets
    │   └── diabetesprediction/
    │       └── images/
    │           ├── mlimage.jpg
    │           └── pro1.jpg
    │
    └── templates/                            # Frontend UI templates
        ├── home.html                         # Home page template
        └── predict.html                      # Interactive risk assessment form template
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: Version 3.10, 3.11, 3.12, or 3.13
- **Git**: Installed and configured on your system

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/unicorn1122/diabetesprediction.git
   cd diabetesprediction
   ```

2. **Create a virtual environment:**
   ```bash
   # Windows (PowerShell / Command Prompt)
   python -m venv .venv
   .venv\Scripts\activate

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running Migrations

Initialize the SQLite database schema:

```bash
cd diabetesprediction
python manage.py migrate
```

### Training / Verifying the Model

You can retrain the model pipeline or verify its serialization at any time:

```bash
# Retrain model pipeline:
python train_model.py
```
*Output will display validation accuracy and classification report on test data.*

### Starting the Development Server

Launch the Django local server:

```bash
python manage.py runserver
```

Open your browser and visit:
- **Home Page**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Prediction Tool**: [http://127.0.0.1:8000/predict/](http://127.0.0.1:8000/predict/)

---

## 🧪 Testing & Validation

The project includes an automated end-to-end workflow verification script:

```bash
python verify_workflow.py
```

This script verifies:
1. HTTP 200 response from the landing page (`/`).
2. HTTP 200 response and form field rendering on the prediction page (`/predict/`).
3. Correct negative classification for non-diabetic clinical profiles.
4. Correct positive classification for elevated risk profiles.
5. Rejection and error messaging for non-numeric, `NaN`, or missing parameters.
6. Display of mandatory medical disclaimer notices.

Alternatively, execute standard Django unit tests:
```bash
python manage.py test
```

---

## ⚠️ Medical & Educational Disclaimer

> **Important Notice:**
> This software is developed strictly for **educational, academic, and research purposes**. It does not constitute medical advice, diagnosis, or clinical recommendation. The predictions generated by the machine learning models reflect statistical correlations within the Pima Indians Diabetes Dataset and should **never** be used as a substitute for professional clinical judgment, laboratory blood testing, or physician consultation.
