import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [col for col in FEATURE_COLUMNS if col not in NUMERIC_FEATURES]

def build_preprocessor():
    """
    Creates a ColumnTransformer to scale numeric features and one-hot encode categorical features.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def train_and_evaluate_models(df):
    """
    Trains Logistic Regression and Random Forest using Scikit-Learn Pipelines with Stratified Splitting.
    Prevents data leakage by fitting preprocessing only on training set.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df['Churn_Numeric'].copy()
    
    # 1. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    preprocessor = build_preprocessor()
    
    # 2. Logistic Regression Pipeline
    log_reg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # 3. Random Forest Pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Fit pipelines on training data only
    log_reg_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)
    
    # Evaluate models on test data
    models = {
        'Logistic Regression': log_reg_pipeline,
        'Random Forest': rf_pipeline
    }
    
    results = {}
    
    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'pipeline': pipeline,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
    # Extract Feature Importances from Random Forest
    rf_clf = rf_pipeline.named_steps['classifier']
    fitted_prep = rf_pipeline.named_steps['preprocessor']
    
    # Get feature names after one-hot encoding
    cat_feature_names = fitted_prep.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = list(NUMERIC_FEATURES) + list(cat_feature_names)
    
    importances = rf_clf.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Clean feature names for clean presentation (e.g. Contract_Month-to-month -> Contract: Month-to-month)
    fi_df['CleanFeature'] = fi_df['Feature'].apply(
        lambda x: x.replace('_', ': ') if '_' in x and not any(num in x for num in NUMERIC_FEATURES) else x
    )
    
    # Best model selection based on F1 Score
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    
    return {
        'models': results,
        'best_model_name': best_model_name,
        'feature_importances': fi_df.head(10),
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape
    }

def predict_single_customer(pipeline, customer_dict):
    """
    Accepts a single customer input dictionary and returns prediction class, probability, and recommendations.
    """
    df_single = pd.DataFrame([customer_dict])
    
    # Ensure types
    df_single['tenure'] = int(df_single['tenure'].iloc[0])
    df_single['MonthlyCharges'] = float(df_single['MonthlyCharges'].iloc[0])
    df_single['TotalCharges'] = float(df_single['TotalCharges'].iloc[0])
    df_single['SeniorCitizen'] = int(df_single['SeniorCitizen'].iloc[0])
    
    pred_class = pipeline.predict(df_single)[0]
    prob_churn = pipeline.predict_proba(df_single)[0][1]
    prob_pct = round(prob_churn * 100, 1)
    
    is_churn = (pred_class == 1 or prob_churn >= 0.5)
    status_label = "LIKELY TO CHURN" if is_churn else "LIKELY TO STAY"
    
    # Retention Recommendations
    recommendations = []
    if is_churn:
        if customer_dict.get('Contract') == 'Month-to-month':
            recommendations.append("📌 <b>Offer Contract Upgrade:</b> Transition customer to a 1-year or 2-year contract with an attractive loyalty discount.")
        if customer_dict.get('TechSupport') == 'No':
            recommendations.append("📌 <b>Free Tech Support Trial:</b> Provide 3 months of free priority Tech Support to improve service satisfaction.")
        if customer_dict.get('PaymentMethod') == 'Electronic check':
            recommendations.append("📌 <b>Automatic Billing Incentive:</b> Encourage switching to automatic bank transfer or credit card with a $5/month billing credit.")
        if float(customer_dict.get('MonthlyCharges', 0)) > 70:
            recommendations.append("📌 <b>Customized Rate Plan:</b> Review usage and offer a tailored promotional discount package.")
        if not recommendations:
            recommendations.append("📌 <b>Proactive Customer Reachout:</b> Assign a retention specialist to conduct a service health check.")
    else:
        recommendations.append("✅ <b>Maintain Satisfaction:</b> Send periodic customer appreciation rewards and maintain current quality of service.")
        
    return {
        'status_label': status_label,
        'churn_probability': prob_pct,
        'is_churn': is_churn,
        'recommendations': recommendations
    }
