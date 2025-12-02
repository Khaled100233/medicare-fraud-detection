=======
# medicare-fraud-detection
=======
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
| **Precision** | 0.XXX |
| **Recall** | 0.XXX |
| **F1-Score** | 0.XXX |
| **ROC-AUC** | 0.XXX |
| **PR-AUC** | 0.XXX |

**Best Model:** [Model Name]

**Business Impact:** $X.XX million estimated savings from fraud prevention

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
- Logistic Regression (interpretability baseline)
- Decision Tree (simple non-linear)
- Random Forest (robust ensemble)
- Gradient Boosting (advanced ensemble)

### 4. Evaluation Strategy
- Train/Validation/Test split (60/20/20)
- Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC
- Cost-benefit analysis
- Comprehensive error analysis with case studies

---

## 🔍 Key Findings

### Most Important Features
1. Total reimbursement amounts (inpatient/outpatient)
2. Number of claims per provider
3. Number of deceased patients
4. Average patient age
5. Unique diagnosis/procedure codes

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

- **Fraud Detection Rate:** XX% of fraudulent providers identified
- **Investigation Efficiency:** Reduced false positives by XX%
- **Estimated Savings:** $X.X million annually
- **ROI:** X:1 return on investigation costs

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

| Member | Responsibilities |
|--------|-----------------|
| [Name 1] | Data exploration, feature engineering |
| [Name 2] | Model development, hyperparameter tuning |
| [Name 3] | Evaluation, error analysis |
| [Name 4] | Documentation, presentation |

---

## 📝 License

This project is for academic purposes as part of the Machine Learning course at German International University of Applied Sciences.

---

## 📧 Contact

For questions or feedback:
- Email: [khaled.ehab@student.giu-uni.de]
- Course: ICS504 Machine Learning, Winter 2025
- Lecturer: Dr. Caroline Sabty

---

## 🙏 Acknowledgments

- Dataset provided by Kaggle
- Centers for Medicare & Medicaid Services (CMS) for domain context
- Course instructors and TAs for guidance

---

**Last Updated:** December 2, 2025
>>>>>>> 5bc523bbaa073604f1b8e0d67b184ba0ed7d4352
