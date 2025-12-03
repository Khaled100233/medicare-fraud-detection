
# medicare-fraud-detection

# Healthcare Provider Fraud Detection

**Team Members:** [Add your team members here]

A machine learning project to detect fraudulent healthcare providers for Medicare using claims data analysis.

---

## 📋 Project Overview

Healthcare fraud costs the U.S. over $68 billion annually. This project develops an intelligent fraud detection system to help the Centers for Medicare & Medicaid Services (CMS) identify high-risk providers using data-driven approaches.

### Key Objectives
- Detect fraudulent providers from multi-table claims data
- Handle severe class imbalance (~10% fraud rate)
- Provide explainable predictions for investigators
- Demonstrate measurable business value

---

## 🎯 Results Summary

| Metric | Score |
|--------|-------|
| **Precision** | 0.5394 |
| **Recall** | 0.8812 |
| **F1-Score** | 0.6692 |
| **ROC-AUC** | 0.9690 |
| **PR-AUC** | 0.7530 |

**Best Model:** Random Forest Classifier

**Business Impact:** $6.72 million estimated net benefit from fraud prevention

**Test Set Performance:**
- Successfully detected **88.1%** of fraudulent providers (89 out of 101)
- **53.9%** precision (of flagged providers, 53.9% are actually fraudulent)
- ROI: **815%** return on investigation costs

---

## 📁 Project Structure

```
fraud_detection_project/
├── README.md
├── requirements.txt
├── data/
│   ├── Train_Beneficiarydata.csv
│   ├── Train_Inpatientdata.csv
│   ├── Train_Outpatientdata.csv
│   ├── Train_labels.csv
│   └── processed_provider_data.csv
├── notebooks/
│   ├── 01_data_exploration_and_feature_engineering.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
├── reports/
│   ├── technical_report.pdf
│   ├── presentation.pptx
│   ├── model_comparison_results.csv
│   ├── feature_importance.csv
│   ├── final_evaluation_report.csv
│   └── figures/
│       ├── target_distribution.png
│       ├── fraud_comparison.png
│       ├── correlation_heatmap.png
│       ├── model_comparison.png
│       ├── roc_pr_curves.png
│       ├── feature_importance.png
│       └── confusion_matrix.png
└── src/
    └── utils.py (optional helper functions)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required packages (see requirements.txt)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-team/fraud_detection_project.git
cd fraud_detection_project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
- Download from: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
- Extract all CSV files to `data/` directory

### Running the Project

Execute notebooks in order:

```bash
jupyter notebook notebooks/01_data_exploration_and_feature_engineering.ipynb
jupyter notebook notebooks/02_modeling.ipynb
jupyter notebook notebooks/03_evaluation.ipynb
```

---

## 📊 Methodology

### 1. Data Exploration & Feature Engineering
- Analyzed 4 interconnected datasets (beneficiaries, inpatient/outpatient claims, labels)
- Created provider-level aggregations from claim-level data
- Engineered 50+ features including financial metrics, patient demographics, and claim patterns

### 2. Class Imbalance Handling
- Applied SMOTE (Synthetic Minority Oversampling Technique)
- Used class weighting in models
- Focused on precision-recall metrics instead of accuracy

### 3. Model Development
Trained and compared 4 models:
- **Logistic Regression** - Precision: 0.442, Recall: 0.802, F1: 0.570
- **Decision Tree** - Precision: 0.462, Recall: 0.753, F1: 0.573
- **Random Forest** ⭐ - Precision: 0.539, Recall: 0.881, F1: 0.669 (Best)
- **Gradient Boosting** - Precision: 0.500, Recall: 0.691, F1: 0.580

### 4. Evaluation Strategy
- Train/Test split (80/20) with stratification
- Validation split from training data (80/20)
- Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC
- Cost-benefit analysis with realistic cost estimates
- Comprehensive error analysis:
  - 76 False Positives (legitimate providers flagged)
  - 12 False Negatives (fraudulent providers missed)
  - Detailed case studies for improvement insights

---

## 🔍 Key Findings

### Most Important Features
1. **Inpatient Maximum Length of Stay** (12.3% importance)
2. **Inpatient Total Length of Stay** (8.8% importance)
3. **Inpatient Average Claims per Beneficiary** (5.8% importance)
4. **Inpatient Number of Beneficiaries** (5.7% importance)
5. **Inpatient Number of Claims** (4.9% importance)
6. **Inpatient Unique Procedures** (4.5% importance)
7. **Inpatient Total Reimbursed Amount** (4.3% importance)
8. **Chronic Conditions Count** (outpatient) (3.1% importance)
9. **Renal Disease Indicator** (3.0% importance)
10. **Outpatient Unique Diagnoses** (2.7% importance)

### Error Analysis Insights

**False Positives (Legitimate flagged as fraud):**
- High-volume specialty practices
- Teaching hospitals with unusual patterns
- Providers treating complex chronic conditions

**False Negatives (Fraud missed):**
- Sophisticated schemes mimicking legitimate behavior
- Low-volume fraudsters operating under radar
- Providers with diversified fraud patterns

### Recommendations
1. Add network analysis features
2. Include temporal pattern detection
3. Implement anomaly detection
4. Regular model retraining with new cases

---

## 💼 Business Impact

- **Fraud Detection Rate:** 88.1% of fraudulent providers identified (89 out of 101)
- **Investigation Efficiency:** 165 providers flagged for investigation (vs. 1,082 total)
- **Estimated Net Benefit:** $6.72 million from fraud prevention
- **ROI:** 8.15:1 return on investigation costs (815% ROI)
- **Cost Savings:**
  - Fraud prevented: $8.9 million (89 caught × $100k each)
  - Investigation cost: $0.825 million (165 investigated × $5k each)
  - Missed fraud cost: $1.2 million (12 missed × $100k each)
  - False positive cost: $0.532 million (76 cases × $7k each)

---

## 📚 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

Install all at once:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

---

## 👥 Team Contributions

| Member | email |
|--------|-----------------|
| Khaled Ehab Attia Hussein | khaled.attia@giu-uni.de |
| Nirvana Saeed | Nirvana.Saeed@giu-uni.de |

---

## 📝 License

This project is for academic purposes as part of the Machine Learning course at German International University of Applied Sciences.

---

## 🙏 Acknowledgments

- Dataset provided by Kaggle
- Centers for Medicare & Medicaid Services (CMS) for domain context
- Course instructors and TAs for guidance

---
