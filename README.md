# 🔧 Predictive Maintenance System using Machine Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

## 📌 Project Overview

Predictive Maintenance has become one of the most important applications of Artificial Intelligence in modern industry.

This project presents a complete Machine Learning pipeline capable of predicting machine failures before they occur using industrial sensor measurements from the **AI4I 2020 Predictive Maintenance Dataset**.

The goal is to reduce unexpected downtime, minimize maintenance costs, and improve equipment reliability by enabling proactive maintenance decisions.

---

# 🚀 Project Features

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning & Preprocessing
- 🔍 Data Leakage Detection
- ⚙️ Feature Engineering
- 🤖 Machine Failure Prediction
- 🌳 Random Forest Classifier
- 📈 Calibrated Logistic Regression
- 📉 Precision-Recall Curve
- 📊 ROC Curve
- 🎯 Recall @ Fixed Precision
- 📋 Feature Importance Analysis
- 💼 Business Impact Evaluation

---

# 📂 Project Structure

```
Predictive-Maintenance-System
│
├── notebook/
│   └── predictive_maintenance.ipynb
│
├── data/
│   └── ai4i2020.csv
│
├── assets/
│   ├── feature_importances_rf.png
│   ├── pr_curve_comparison.png
│   ├── metrics_summary.csv
│   └── operating_point_uplift.csv
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# 📊 Dataset

Dataset Used:

**AI4I 2020 Predictive Maintenance Dataset**

The dataset contains approximately **10,000 industrial machine records** with sensor measurements including:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Failure Label

Target Variable:

**Machine Failure (Binary Classification)**

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Jupyter Notebook

---

# 🤖 Machine Learning Models

### Random Forest Classifier

- Non-linear classification
- Feature Importance
- Robust against overfitting

### Logistic Regression (Calibrated)

- Probability Calibration
- High Interpretability
- Balanced Class Weight

---

# 📈 Model Evaluation

The project evaluates models using multiple metrics:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Recall @ 90% Precision

---

# 📊 Results

| Model | Description |
|--------|------------|
| Random Forest | Strong baseline with high predictive performance |
| Logistic Regression | Interpretable calibrated model |

Performance reports are automatically saved inside:

```
assets/
```

Including:

- metrics_summary.csv
- operating_point_uplift.csv
- feature_importances_rf.png
- pr_curve_comparison.png

---

# 📸 Project Outputs

### Precision-Recall Curve

![PR Curve](assets/pr_curve_comparison.png)

---

### Feature Importance

![Feature Importance](assets/feature_importances_rf.png)

---

# 💡 Business Impact

The developed model enables maintenance teams to:

- Predict machine failures before breakdowns
- Reduce unplanned downtime
- Improve maintenance scheduling
- Lower operational costs
- Increase equipment reliability

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/HBE-2206/Predictive-Maintenance-System.git
```

Move into the project

```bash
cd Predictive-Maintenance-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Run

```
predictive_maintenance.ipynb
```

---

# 🔮 Future Improvements

- XGBoost
- LightGBM
- CatBoost
- Hyperparameter Optimization
- SHAP Explainability
- Streamlit Web Application
- Flask REST API
- Docker Deployment

---

# 👨‍💻 Author

## Hazem Mohamed

Computer Science & Artificial Intelligence Student

Machine Learning Engineer

ROV Team Leader

📧 Email: Your Email

💼 LinkedIn:
https://linkedin.com/in/YourProfile

🐙 GitHub:
https://github.com/HBE-2206

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.