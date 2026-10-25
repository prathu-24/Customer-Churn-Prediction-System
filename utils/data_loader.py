import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

def generate_sample_telco_data(num_rows=7043):
    """
    Generates a realistic Telco Customer Churn dataset matching IBM Telco specifications
    in case the CSV file is not locally present.
    """
    np.random.seed(42)
    
    customer_ids = [f"{np.random.randint(1000, 9999)}-{np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5)}" for _ in range(num_rows)]
    genders = np.random.choice(['Female', 'Male'], size=num_rows, p=[0.495, 0.505])
    senior_citizens = np.random.choice([0, 1], size=num_rows, p=[0.838, 0.162])
    partners = np.random.choice(['Yes', 'No'], size=num_rows, p=[0.483, 0.517])
    dependents = np.random.choice(['Yes', 'No'], size=num_rows, p=[0.300, 0.700])
    
    # Tenure (months) between 1 and 72
    tenures = np.random.randint(1, 73, size=num_rows)
    # Give some brand new customers 0 tenure for realistic blank TotalCharges
    zero_indices = np.random.choice(num_rows, size=11, replace=False)
    tenures[zero_indices] = 0
    
    phone_service = np.random.choice(['Yes', 'No'], size=num_rows, p=[0.903, 0.097])
    multiple_lines = []
    for ps in phone_service:
        if ps == 'No':
            multiple_lines.append('No phone service')
        else:
            multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.46, 0.54]))
            
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=num_rows, p=[0.34, 0.44, 0.22])
    
    online_sec, online_bk, dev_prot, tech_sup, stream_tv, stream_mov = [], [], [], [], [], []
    for inet in internet_service:
        if inet == 'No':
            online_sec.append('No internet service')
            online_bk.append('No internet service')
            dev_prot.append('No internet service')
            tech_sup.append('No internet service')
            stream_tv.append('No internet service')
            stream_mov.append('No internet service')
        else:
            online_sec.append(np.random.choice(['Yes', 'No'], p=[0.29, 0.71]))
            online_bk.append(np.random.choice(['Yes', 'No'], p=[0.34, 0.66]))
            dev_prot.append(np.random.choice(['Yes', 'No'], p=[0.34, 0.66]))
            tech_sup.append(np.random.choice(['Yes', 'No'], p=[0.29, 0.71]))
            stream_tv.append(np.random.choice(['Yes', 'No'], p=[0.39, 0.61]))
            stream_mov.append(np.random.choice(['Yes', 'No'], p=[0.39, 0.61]))
            
    contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_rows, p=[0.55, 0.21, 0.24])
    paperless = np.random.choice(['Yes', 'No'], size=num_rows, p=[0.59, 0.41])
    pmt_methods = np.random.choice([
        'Electronic check', 'Mailed check', 
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ], size=num_rows, p=[0.33, 0.23, 0.22, 0.22])
    
    # Realistic monthly charges estimation
    monthly_charges = []
    for i in range(num_rows):
        base = 20.0
        if phone_service[i] == 'Yes': base += 10.0
        if multiple_lines[i] == 'Yes': base += 15.0
        if internet_service[i] == 'DSL': base += 25.0
        elif internet_service[i] == 'Fiber optic': base += 45.0
        if online_sec[i] == 'Yes': base += 8.0
        if online_bk[i] == 'Yes': base += 8.0
        if dev_prot[i] == 'Yes': base += 8.0
        if tech_sup[i] == 'Yes': base += 8.0
        if stream_tv[i] == 'Yes': base += 10.0
        if stream_mov[i] == 'Yes': base += 10.0
        # Add slight noise
        base += np.random.uniform(-3, 3)
        monthly_charges.append(round(max(18.25, base), 2))
        
    monthly_charges = np.array(monthly_charges)
    
    total_charges = []
    for i in range(num_rows):
        if tenures[i] == 0:
            total_charges.append(" ") # Simulate empty string in original dataset
        else:
            tc = monthly_charges[i] * tenures[i] + np.random.uniform(-10, 10)
            total_charges.append(str(round(max(18.85, tc), 2)))
            
    # Realistic churn probability calculation
    churn_labels = []
    for i in range(num_rows):
        prob = 0.20
        if contracts[i] == 'Month-to-month': prob += 0.25
        elif contracts[i] == 'Two year': prob -= 0.15
        
        if internet_service[i] == 'Fiber optic': prob += 0.15
        if tech_sup[i] == 'No': prob += 0.10
        if online_sec[i] == 'No': prob += 0.10
        if pmt_methods[i] == 'Electronic check': prob += 0.10
        if tenures[i] < 12: prob += 0.15
        elif tenures[i] > 36: prob -= 0.15
        
        prob = max(0.05, min(0.90, prob))
        churn_labels.append('Yes' if np.random.rand() < prob else 'No')
        
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': genders,
        'SeniorCitizen': senior_citizens,
        'Partner': partners,
        'Dependents': dependents,
        'tenure': tenures,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_sec,
        'OnlineBackup': online_bk,
        'DeviceProtection': dev_prot,
        'TechSupport': tech_sup,
        'StreamingTV': stream_tv,
        'StreamingMovies': stream_mov,
        'Contract': contracts,
        'PaperlessBilling': paperless,
        'PaymentMethod': pmt_methods,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn_labels
    })
    
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df

def get_data_filepath():
    return DATA_PATH

def load_raw_dataset():
    """
    Loads raw CSV dataset from data folder.
    Creates sample Telco dataset automatically if missing.
    """
    if not os.path.exists(DATA_PATH):
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as response:
                df = pd.read_csv(response)
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            df.to_csv(DATA_PATH, index=False)
        except Exception:
            df = generate_sample_telco_data()
    else:
        df = pd.read_csv(DATA_PATH)
    return df


def clean_dataset(df):
    """
    Cleans raw dataset:
    - Converts TotalCharges to numeric (handles blanks/spaces gracefully)
    - Fills missing TotalCharges values with 0
    - Converts Churn column to 0/1 numeric binary
    - Returns cleaned DataFrame
    """
    df_clean = df.copy()
    
    # 1. TotalCharges conversion to numeric
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
    
    # 2. Ensure tenure is integer
    df_clean['tenure'] = df_clean['tenure'].astype(int)
    
    # 3. Numeric Churn indicator
    if 'Churn' in df_clean.columns:
        df_clean['Churn_Numeric'] = df_clean['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
        
    return df_clean
