# 🛡️ Hybrid AI/ML Network Intrusion Detection System with Explainable AI (XAI)

## AI and Machine Learning Techniques for Data Privacy and Security: Bridging Legal Requirements with Technical Solutions Across Network Domains

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE](https://img.shields.io/badge/Format-IEEE-orange.svg)](https://www.ieee.org)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Dataset Information](#dataset-information)
4. [System Requirements](#system-requirements)
5. [Installation](#installation)
6. [Project Structure](#project-structure)
7. [Usage Guide](#usage-guide)
8. [Model Performance](#model-performance)
9. [Explainable AI (XAI)](#explainable-ai-xai)
10. [Legal-Technical Alignment Framework](#legal-technical-alignment-framework)
11. [Results and Visualizations](#results-and-visualizations)
12. [References](#references)
13. [Citation](#citation)
14. [License](#license)
15. [Contact](#contact)

---

## 📖 Project Overview

This project implements a **Hybrid AI/ML-based Network Intrusion Detection System (IDS)** that combines multiple machine learning paradigms with **Explainable AI (XAI)** capabilities for transparent, interpretable security decisions.

### 🎯 Objectives

1. **Implement** a hybrid AI/ML-based IDS combining multiple learning paradigms
2. **Integrate** Explainable AI (XAI) techniques for GDPR Article 22 compliance
3. **Demonstrate** the Legal-Technical Alignment Framework (LTAF) in practice
4. **Achieve** high detection accuracy while maintaining model interpretability

### 🏛️ Academic Context

- **Module:** INF613 - Computer Network and Data Security
- **Institution:** The British University in Dubai
- **Academic Year:** 2025-26
- **Assignment:** Implementation and Evaluation of Contemporary AI/ML Technique

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🌲 **Random Forest** | Ensemble learning with 150 decision trees |
| 🚀 **XGBoost** | Gradient boosting for high accuracy |
| 🧠 **Deep Neural Network** | 4-layer MLP for complex pattern recognition |
| 🗳️ **Voting Ensemble** | Soft voting combining all models |
| 🔍 **SHAP Explanations** | Feature-level interpretability for each prediction |
| ⚖️ **GDPR Compliance** | Addresses Article 22 "right to explanation" |
| 📊 **Comprehensive Evaluation** | Multiple metrics with cross-validation |

---

## 📊 Dataset Information

### NSL-KDD Dataset

The **NSL-KDD** dataset is used for training and evaluation. It is a refined version of the original KDD Cup 1999 dataset, developed by the **Canadian Institute for Cybersecurity** at the University of New Brunswick.

#### 📥 Dataset Source

- **Official URL:** https://www.unb.ca/cic/datasets/nsl.html
- **Alternative:** Kaggle, UCI Machine Learning Repository

#### 📈 Dataset Statistics

| Property | Training Set | Test Set |
|----------|-------------|----------|
| **Total Samples** | 25,192 | 22,544 |
| **Features** | 41 | 41 |
| **Normal Class** | 13,449 (53.4%) | - |
| **Anomaly Class** | 11,743 (46.6%) | - |
| **Missing Values** | 0 | 0 |

#### 🏷️ Feature Categories

The 41 features are organized into **4 categories**:

##### 1. Basic Features (9 features)
These describe individual TCP connections:

| Feature | Type | Description |
|---------|------|-------------|
| `duration` | Continuous | Length of connection (seconds) |
| `protocol_type` | Categorical | Protocol type (TCP, UDP, ICMP) |
| `service` | Categorical | Network service (http, ftp, smtp, etc.) |
| `flag` | Categorical | Connection status flag |
| `src_bytes` | Continuous | Bytes sent from source to destination |
| `dst_bytes` | Continuous | Bytes sent from destination to source |
| `land` | Binary | 1 if connection is from/to same host/port |
| `wrong_fragment` | Continuous | Number of wrong fragments |
| `urgent` | Continuous | Number of urgent packets |

##### 2. Content Features (13 features)
Derived from domain knowledge about attacks:

| Feature | Type | Description |
|---------|------|-------------|
| `hot` | Continuous | Number of "hot" indicators |
| `num_failed_logins` | Continuous | Number of failed login attempts |
| `logged_in` | Binary | 1 if successfully logged in |
| `num_compromised` | Continuous | Number of compromised conditions |
| `root_shell` | Binary | 1 if root shell obtained |
| `su_attempted` | Binary | 1 if "su root" command attempted |
| `num_root` | Continuous | Number of root accesses |
| `num_file_creations` | Continuous | Number of file creation operations |
| `num_shells` | Continuous | Number of shell prompts |
| `num_access_files` | Continuous | Number of access control files |
| `num_outbound_cmds` | Continuous | Number of outbound commands (always 0) |
| `is_host_login` | Binary | 1 if login belongs to "hot" list |
| `is_guest_login` | Binary | 1 if login is a guest login |

##### 3. Time-Based Traffic Features (9 features)
Computed over a 2-second time window:

| Feature | Type | Description |
|---------|------|-------------|
| `count` | Continuous | Connections to same host in past 2 sec |
| `srv_count` | Continuous | Connections to same service in past 2 sec |
| `serror_rate` | Continuous | % connections with SYN errors |
| `srv_serror_rate` | Continuous | % connections to same service with SYN errors |
| `rerror_rate` | Continuous | % connections with REJ errors |
| `srv_rerror_rate` | Continuous | % connections to same service with REJ errors |
| `same_srv_rate` | Continuous | % connections to same service |
| `diff_srv_rate` | Continuous | % connections to different services |
| `srv_diff_host_rate` | Continuous | % connections to different hosts |

##### 4. Host-Based Traffic Features (10 features)
Computed over 100 connections to same destination host:

| Feature | Type | Description |
|---------|------|-------------|
| `dst_host_count` | Continuous | Connections to same destination host |
| `dst_host_srv_count` | Continuous | Connections to same service on dest host |
| `dst_host_same_srv_rate` | Continuous | % connections to same service |
| `dst_host_diff_srv_rate` | Continuous | % connections to different services |
| `dst_host_same_src_port_rate` | Continuous | % connections from same source port |
| `dst_host_srv_diff_host_rate` | Continuous | % connections to different hosts |
| `dst_host_serror_rate` | Continuous | % connections with SYN errors |
| `dst_host_srv_serror_rate` | Continuous | % connections to same service with SYN errors |
| `dst_host_rerror_rate` | Continuous | % connections with REJ errors |
| `dst_host_srv_rerror_rate` | Continuous | % connections to same service with REJ errors |

#### 🎯 Target Variable

| Class | Description | Count | Percentage |
|-------|-------------|-------|------------|
| `normal` | Legitimate network traffic | 13,449 | 53.4% |
| `anomaly` | Malicious/attack traffic | 11,743 | 46.6% |

#### ⚔️ Attack Categories (in Anomaly Class)

The anomaly class encompasses four major attack categories:

| Category | Description | Examples |
|----------|-------------|----------|
| **DoS** | Denial of Service | neptune, smurf, pod, teardrop, land, back |
| **Probe** | Surveillance/Scanning | portsweep, ipsweep, nmap, satan |
| **R2L** | Remote to Local | ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster |
| **U2R** | User to Root | buffer_overflow, loadmodule, perl, rootkit |

#### 📝 Categorical Feature Values

```
protocol_type: ['tcp', 'udp', 'icmp'] (3 unique values)

service: ['http', 'smtp', 'finger', 'domain_u', 'auth', 'telnet', 'ftp', 
          'ftp_data', 'other', 'private', 'remote_job', 'name', 'netbios_ns',
          'eco_i', 'mtp', 'link', 'supdup', 'gopher', 'rje', 'sql_net',
          'vmnet', 'csnet_ns', 'pop_2', 'nntp', 'imap4', 'time', 'netbios_dgm',
          'ssh', 'http_443', 'login', 'exec', 'shell', ...] (70 unique values)

flag: ['SF', 'S1', 'REJ', 'S2', 'S0', 'S3', 'RSTO', 'RSTR', 'RSTOS0', 
       'OTH', 'SH'] (11 unique values)
```

#### 🔄 Data Preprocessing Applied

1. **Label Encoding:** Categorical features converted to numerical
2. **Standard Scaling:** Zero mean, unit variance normalization
3. **SMOTE:** Synthetic Minority Over-sampling for class balance
4. **Train-Val Split:** 80-20 stratified split

---

## 💻 System Requirements

### Minimum Requirements

- **Python:** 3.8 or higher
- **RAM:** 8 GB minimum (16 GB recommended)
- **Storage:** 2 GB free space
- **OS:** Windows 10/11, macOS 10.14+, Ubuntu 18.04+

### Required Libraries

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0
shap>=0.40.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
tqdm>=4.62.0
```

---

## 🚀 Installation

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hybrid-ids-xai.git
cd hybrid-ids-xai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the main implementation
python main_ids_implementation.py
```

### Option 2: Google Colab

1. Upload `Hybrid_IDS_XAI_Network_Intrusion_Detection.ipynb` to Google Colab
2. Upload `Train_data.csv` and `Test_data.csv` when prompted
3. Run all cells sequentially

---

## 📁 Project Structure

```
hybrid-ids-xai/
│
├── 📄 README.md                           # This file
├── 📄 requirements.txt                    # Python dependencies
├── 📄 LICENSE                             # MIT License
│
├── 📓 Hybrid_IDS_XAI_Network_Intrusion_Detection.ipynb  # Google Colab notebook
├── 🐍 main_ids_implementation.py          # Main Python implementation
│
├── 📊 data/
│   ├── Train_data.csv                     # Training dataset (NSL-KDD)
│   └── Test_data.csv                      # Test dataset
│
├── 📄 reports/
│   └── IEEE_Full_Report_Hybrid_IDS_XAI.docx  # Complete IEEE format report
│
└── 📈 results/
    ├── figures/
    │   ├── model_comparison.png           # Performance comparison chart
    │   ├── confusion_matrices.png         # Confusion matrices for all models
    │   ├── roc_curves.png                 # ROC curves comparison
    │   ├── feature_importance.png         # RF and SHAP feature importance
    │   ├── shap_summary.png               # SHAP summary plot
    │   └── precision_recall.png           # Precision-recall curves
    │
    ├── models/
    │   ├── random_forest.joblib           # Trained Random Forest model
    │   ├── xgboost.joblib                 # Trained XGBoost model
    │   ├── deep_neural_network.joblib     # Trained DNN model
    │   ├── voting_ensemble.joblib         # Trained Ensemble model
    │   ├── scaler.joblib                  # StandardScaler
    │   └── label_encoders.joblib          # Label encoders
    │
    └── report_data.json                   # Results in JSON format
```

---

## 📖 Usage Guide

### Quick Start

```python
# Import the implementation
from main_ids_implementation import HybridIDS

# Initialize the system
ids = HybridIDS(random_state=42)

# Load and preprocess data
ids.load_data('Train_data.csv')
ids.preprocess()

# Train all models
ids.train_models()

# Evaluate performance
results = ids.evaluate()

# Generate SHAP explanations
ids.explain_predictions()

# Save models
ids.save_models('results/models/')
```

### Using Pre-trained Models

```python
import joblib
import numpy as np

# Load models
xgb_model = joblib.load('results/models/xgboost.joblib')
scaler = joblib.load('results/models/scaler.joblib')
encoders = joblib.load('results/models/label_encoders.joblib')

# Preprocess new data
# ... (apply same preprocessing)

# Make predictions
predictions = xgb_model.predict(X_new_scaled)
probabilities = xgb_model.predict_proba(X_new_scaled)

print(f"Prediction: {'Anomaly' if predictions[0] == 1 else 'Normal'}")
print(f"Confidence: {max(probabilities[0]) * 100:.2f}%")
```

### Generating SHAP Explanations

```python
import shap

# Load model and create explainer
xgb_model = joblib.load('results/models/xgboost.joblib')
explainer = shap.TreeExplainer(xgb_model)

# Generate SHAP values
shap_values = explainer.shap_values(X_sample)

# Visualize
shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
```

---

## 📊 Model Performance

### Classification Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 99.72% | 99.72% | 99.72% | 99.72% | 1.0000 |
| **XGBoost** | **99.80%** | **99.80%** | **99.80%** | **99.80%** | **1.0000** |
| Deep Neural Network | 99.46% | 99.46% | 99.46% | 99.46% | 0.9997 |
| Voting Ensemble | 99.80% | 99.80% | 99.80% | 99.80% | 1.0000 |

### Cross-Validation Results (5-Fold)

| Model | Mean Accuracy | Std Deviation |
|-------|---------------|---------------|
| XGBoost | 99.75% | ±0.09% |
| Random Forest | 99.69% | ±0.10% |
| Deep Neural Network | 99.17% | ±0.38% |

### Confusion Matrix (XGBoost - Best Model)

|  | Predicted Normal | Predicted Anomaly |
|--|------------------|-------------------|
| **Actual Normal** | 2,342 (TN) | 7 (FP) |
| **Actual Anomaly** | 3 (FN) | 2,687 (TP) |

- **True Positive Rate:** 99.89%
- **False Positive Rate:** 0.30%
- **False Negative Rate:** 0.11%

---

## 🔍 Explainable AI (XAI)

### Top 10 Most Important Features (SHAP Analysis)

| Rank | Feature | SHAP Value | Interpretation |
|------|---------|------------|----------------|
| 1 | `src_bytes` | 3.4095 | Bytes sent from source - indicates data exfiltration |
| 2 | `dst_bytes` | 1.1900 | Bytes received - reveals abnormal response patterns |
| 3 | `count` | 0.9513 | Connection frequency - detects scanning attacks |
| 4 | `dst_host_same_src_port_rate` | 0.6238 | Port pattern - identifies port scanning |
| 5 | `dst_host_srv_count` | 0.6211 | Service connections - reveals service abuse |
| 6 | `dst_host_same_srv_rate` | 0.5797 | Service consistency - detects anomalies |
| 7 | `protocol_type` | 0.4166 | Protocol used - attack-specific patterns |
| 8 | `hot` | 0.4010 | Hot indicators - suspicious activities |
| 9 | `service` | 0.3807 | Service type - service-specific attacks |
| 10 | `dst_host_rerror_rate` | 0.3352 | Error rate - connection failures |

### Why SHAP?

SHAP (SHapley Additive exPlanations) provides:

- ✅ **Local Interpretability:** Explains individual predictions
- ✅ **Global Interpretability:** Shows overall feature importance
- ✅ **Mathematical Consistency:** Based on game theory (Shapley values)
- ✅ **Model Agnostic:** Works with any ML model
- ✅ **GDPR Compliance:** Addresses Article 22 "right to explanation"

---

## ⚖️ Legal-Technical Alignment Framework (LTAF)

This implementation demonstrates compliance with privacy regulations:

| Legal Principle | Regulation | Technical Implementation |
|-----------------|------------|--------------------------|
| **Transparency** | GDPR Art. 22 | SHAP explanations, Feature importance rankings |
| **Accountability** | GDPR Art. 5(2) | Model versioning, Prediction logging, Audit trails |
| **Security-by-Design** | GDPR Art. 25 | Real-time IDS, Multi-model ensemble redundancy |
| **Data Minimization** | GDPR Art. 5(1c) | Feature selection based on importance analysis |
| **Accuracy** | GDPR Art. 5(1d) | Cross-validation, Multi-metric evaluation |

---

## 📈 Results and Visualizations

All visualizations are saved in `results/figures/`:

1. **model_comparison.png** - Bar chart comparing all metrics across models
2. **confusion_matrices.png** - 2x2 grid of confusion matrices
3. **roc_curves.png** - ROC curves with AUC values
4. **feature_importance.png** - Random Forest vs SHAP importance
5. **shap_summary.png** - SHAP summary plot (beeswarm)
6. **precision_recall.png** - Precision-Recall curves

---

## 📚 References

1. Tavallaee, M., et al. (2009). "A detailed analysis of the KDD CUP 99 data set." IEEE CISDA.
2. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." NeurIPS.
3. Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." ACM SIGKDD.
4. European Union. (2016). "General Data Protection Regulation (GDPR)."
5. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." JAIR.

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{hybrid_ids_xai_2025,
  title={Hybrid AI/ML Network Intrusion Detection with Explainable AI},
  author={[Your Name]},
  institution={The British University in Dubai},
  year={2025},
  note={INF613 - Computer Network and Data Security}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

- **Student:** [Your Name]
- **Student ID:** [Your ID]
- **Email:** [Your Email]
- **Institution:** The British University in Dubai
- **Module:** INF613 - Computer Network and Data Security

---

## 🙏 Acknowledgments

- Canadian Institute for Cybersecurity (NSL-KDD Dataset)
- SHAP Library Contributors
- Scikit-learn and XGBoost Development Teams
- Module Tutor: Dr. Suleiman Yerima

---

**⭐ If this project helped you, please consider giving it a star!**
