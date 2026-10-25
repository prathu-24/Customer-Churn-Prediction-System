# Customer Churn Analysis & Prediction

A complete, modern, professional Machine Learning web application built using **Python**, **Streamlit**, **Pandas**, **NumPy**, **Scikit-learn**, and **Plotly** to analyze customer retention data, discover key churn factors, benchmark classification models, and predict customer churn risk in real time.

---

## 📌 Project Objectives

1. **Quantify Churn:** Measure overall churn count, churn percentage rate, and financial impact.
2. **Discover Factors:** Identify demographic, contract, financial, and service factors associated with customer churn.
3. **Compare Machine Learning Models:** Train and evaluate **Logistic Regression** and **Random Forest Classifier** models without data leakage.
4. **Interactive Prediction & Recommendations:** Allow users to input individual customer attributes and receive immediate churn probabilities along with tailored retention strategies.
5. **Model Explainability:** Visualize feature importance weights to explain model predictions for academic viva defense.

---

## 🛠️ Technologies & Libraries Used

- **Language:** Python 3.10+
- **Web Interface:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Pipelines, ColumnTransformer, LogisticRegression, RandomForestClassifier)
- **Data Visualization:** Plotly Express & Plotly Graph Objects
- **Model Persistence:** Joblib

---

## 📊 Dataset Description

The application utilizes the standard **IBM Telco Customer Churn Dataset** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).

### Features Handled:
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Account & Financial:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Target Variable:** `Churn` (Converted to binary 0 = Retained, 1 = Churned)

> **Note:** If the CSV file is missing from the `data/` folder, the application will automatically fetch or generate a realistic, standard dataset so it works completely out of the box.

---

## 🔄 Project Workflow Pipeline

```
Customer Data (CSV) 
       │
       ▼
Data Cleaning (Missing Values & Type Conversion)
       │
       ▼
Exploratory Data Analysis (10 Plotly Charts)
       │
       ▼
Feature Engineering & Pipelines (StandardScaler + OneHotEncoder)
       │
       ▼
Model Training & Evaluation (Logistic Regression vs Random Forest)
       │
       ▼
Real-Time GUI Churn Prediction & Retention Recommendations
```

---

## 🚀 Installation & Setup Guide

### 1. Clone or Download Project
Ensure all project files are located in your workspace directory:
```
customer_churn_project/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── utils/
    ├── __init__.py
    ├── data_loader.py
    ├── eda_charts.py
    ├── insights_engine.py
    └── model_manager.py
```

### 2. Install Required Dependencies
Open your terminal/command prompt in the project root directory and run:
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
Run the following command:
```bash
streamlit run app.py
```

After executing the command, Streamlit will open the interactive dashboard in your default browser at `http://localhost:8501`.

---

## 🖥️ Application Features & Navigation Pages

1. **🏠 Home / Overview:** Dashboard header with KPI cards (Total Customers, Churned Customers, Churn Rate, Avg Monthly Charges) and interactive workflow pipeline.
2. **📊 Dataset Explorer:** Preview dataset table, inspect schema types, filter records, view missing values, and review descriptive statistics.
3. **📈 Exploratory Data Analysis (EDA):** 10 interactive Plotly charts visualizing churn against Contract types, Tenure, Monthly Charges, Tech Support, Internet Service, etc.
4. **💡 Customer Churn Insights:** Automatically calculated statistical observations and high-risk factor summary tables.
5. **🤖 Machine Learning Model:** Benchmark comparison table (Accuracy, Precision, Recall, F1 Score) and Confusion Matrices for Logistic Regression and Random Forest.
6. **🔮 Churn Prediction:** Interactive input form for real-time customer churn prediction with probability progress bar and custom retention strategies.
7. **🔍 Model Explainability:** Horizontal Plotly bar chart displaying the top 10 feature importances from Random Forest.
8. **ℹ️ About Project:** Project overview and curated viva defense Q&A guide.

---

## 🎓 Viva Defense Prep (Sample Questions & Answers)

- **Q: What is Data Leakage and how did you avoid it?**  
  *A: Data leakage occurs when information from the test set influences model training. We avoided it by wrapping preprocessing (`StandardScaler` and `OneHotEncoder`) inside Scikit-learn `Pipeline` objects and fitting them strictly on the training set.*
- **Q: Why use F1 Score instead of Accuracy for model selection?**  
  *A: In churn prediction, dataset classes are often imbalanced. F1 Score balances Precision and Recall, ensuring the model accurately identifies true churners without excessive false alarms.*

---

## 🔮 Future Enhancements

- Integrate advanced gradient boosting models (XGBoost, LightGBM).
- Export customer churn prediction reports as downloadable PDFs.
- Add live database connectivity (SQL/MongoDB) for streaming enterprise customer data.
