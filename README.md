# Customer Churn Analysis & Prediction

A Machine Learning web application built using **Python, Streamlit, Pandas, NumPy, Scikit-learn, and Plotly** to analyze customer retention, discover churn factors, compare classification models, and predict customer churn risk.

## 📌 Project Objectives

1. Measure customer churn count, churn rate, and financial impact.
2. Identify demographic, contract, financial, and service factors associated with churn.
3. Compare Logistic Regression and Random Forest classification models.
4. Predict customer churn probability and provide retention recommendations.
5. Visualize feature importance to explain model predictions.

## 🛠️ Technologies Used

- **Language:** Python 3.10+
- **Web Interface:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Plotly Express and Plotly Graph Objects
- **Model Persistence:** Joblib

## 📊 Dataset

This project uses the IBM Telco Customer Churn dataset.

**Target variable:** `Churn`
- `0` — Customer retained
- `1` — Customer churned

Features include customer demographics, internet and phone services, contract details, tenure, payment methods, and monthly charges.

## 🔄 Project Workflow

1. Load and clean customer data.
2. Perform exploratory data analysis using interactive charts.
3. Prepare numerical and categorical features.
4. Train and evaluate Logistic Regression and Random Forest models.
5. Predict customer churn and generate retention recommendations.
6. Visualize important features influencing predictions.

## 🚀 Installation and Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
streamlit run app.py
```

The application should open in your browser at `http://localhost:8501`.

## 🖥️ Application Features

1. **Home / Overview:** Customer statistics and churn KPIs.
2. **Dataset Explorer:** Dataset preview, filtering, and descriptive statistics.
3. **Exploratory Data Analysis:** Interactive charts showing churn patterns.
4. **Customer Churn Insights:** Statistical observations and high-risk factors.
5. **Machine Learning Models:** Model comparison and evaluation metrics.
6. **Churn Prediction:** Interactive customer inputs and churn probability.
7. **Model Explainability:** Feature importance visualization.
8. **About Project:** Project information and viva preparation.

## 🎓 Sample Viva Questions

**Q1. What is data leakage?**

Data leakage occurs when information from the test data influences model training. Using a Scikit-learn pipeline helps ensure preprocessing is fitted on the training data only.

**Q2. Why is F1 Score useful for churn prediction?**

F1 Score balances precision and recall. This is useful when the number of customers who churn is different from the number who stay.

## 🔮 Future Enhancements

- Integrate advanced models such as XGBoost and LightGBM.
- Add downloadable prediction reports.
- Integrate a database for storing customer information.
