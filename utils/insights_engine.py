import pandas as pd

def generate_dynamic_insights(df):
    """
    Computes key statistical churn insights directly from the dataset.
    Returns a dictionary of structured metrics and observational bullet points.
    """
    total_customers = len(df)
    churned_df = df[df['Churn'] == 'Yes']
    retained_df = df[df['Churn'] == 'No']
    
    churn_count = len(churned_df)
    overall_churn_rate = (churn_count / total_customers * 100) if total_customers > 0 else 0
    
    # 1. Contract with highest churn rate
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
    top_contract = contract_churn.idxmax()
    top_contract_rate = contract_churn.max()
    
    # 2. Internet Service with highest churn rate
    internet_churn = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
    top_internet = internet_churn.idxmax()
    top_internet_rate = internet_churn.max()
    
    # 3. Average Monthly Charges
    avg_monthly_churned = churned_df['MonthlyCharges'].mean() if churn_count > 0 else 0
    avg_monthly_retained = retained_df['MonthlyCharges'].mean() if len(retained_df) > 0 else 0
    
    # 4. Average Tenure
    avg_tenure_churned = churned_df['tenure'].mean() if churn_count > 0 else 0
    avg_tenure_retained = retained_df['tenure'].mean() if len(retained_df) > 0 else 0
    
    # 5. Payment Method highest churn
    pmt_churn = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
    top_pmt = pmt_churn.idxmax()
    top_pmt_rate = pmt_churn.max()
    
    # 6. Tech Support effect
    tech_churn = df.groupby('TechSupport')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
    tech_yes_rate = tech_churn.get('Yes', 0)
    tech_no_rate = tech_churn.get('No', 0)
    
    insights = {
        'total_customers': total_customers,
        'churn_count': churn_count,
        'overall_churn_rate': round(overall_churn_rate, 2),
        'top_contract': top_contract,
        'top_contract_rate': round(top_contract_rate, 2),
        'top_internet': top_internet,
        'top_internet_rate': round(top_internet_rate, 2),
        'avg_monthly_churned': round(avg_monthly_churned, 2),
        'avg_monthly_retained': round(avg_monthly_retained, 2),
        'avg_tenure_churned': round(avg_tenure_churned, 1),
        'avg_tenure_retained': round(avg_tenure_retained, 1),
        'top_pmt': top_pmt,
        'top_pmt_rate': round(top_pmt_rate, 2),
        'tech_yes_rate': round(tech_yes_rate, 2),
        'tech_no_rate': round(tech_no_rate, 2),
        'observations': [
            f"<b>Overall Churn Rate:</b> {overall_churn_rate:.1f}% of total customers have discontinued service.",
            f"<b>Contract Vulnerability:</b> <b>{top_contract}</b> customers exhibit the highest churn rate at <b>{top_contract_rate:.1f}%</b>, compared to long-term contract holders.",
            f"<b>Internet Service Risk:</b> Customers subscribed to <b>{top_internet}</b> experience a churn rate of <b>{top_internet_rate:.1f}%</b>.",
            f"<b>Financial Sensitivity:</b> Churned customers have a higher average monthly charge (<b>${avg_monthly_churned:.2f}</b>) than retained customers (<b>${avg_monthly_retained:.2f}</b>).",
            f"<b>Tenure & Loyalty:</b> Customers who churn stay for an average of <b>{avg_tenure_churned:.1f} months</b>, whereas retained customers average <b>{avg_tenure_retained:.1f} months</b>.",
            f"<b>Payment Preference:</b> The <b>{top_pmt}</b> payment method is associated with the highest customer churn (<b>{top_pmt_rate:.1f}%</b>).",
            f"<b>Support Value:</b> Customers with Tech Support have a much lower churn rate (<b>{tech_yes_rate:.1f}%</b>) than those without Tech Support (<b>{tech_no_rate:.1f}%</b>)."
        ]
    }
    
    return insights
