import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from utils.data_loader import load_raw_dataset, clean_dataset, get_data_filepath
from utils.eda_charts import (
    plot_churn_distribution, plot_churn_by_gender, plot_churn_by_contract,
    plot_churn_by_internet, plot_churn_by_payment, plot_churn_by_senior,
    plot_churn_by_tech_support, plot_churn_by_tenure,
    plot_monthly_charges_vs_churn, plot_total_charges_vs_churn
)
from utils.insights_engine import generate_dynamic_insights
from utils.model_manager import train_and_evaluate_models, predict_single_customer

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Analysis & Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Modern Professional Styling & Top Navigation Navbar
st.markdown("""
<style>
    /* Custom Styling for Top Navigation Navbar Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0F172A;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 25px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        color: #94A3B8 !important;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0 18px;
        background-color: transparent;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1E293B;
        color: #F8FAFC !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Card Container Styling */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-card h4 {
        color: #94A3B8 !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .metric-card .value {
        color: #38BDF8 !important;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Workflow Pipeline Steps */
    .workflow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        margin: 25px 0;
    }
    .workflow-step {
        background: #1E293B;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 15px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        flex: 1;
        min-width: 140px;
        text-align: center;
        font-weight: 600;
        color: #F8FAFC !important;
    }
    .workflow-arrow {
        color: #94A3B8;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    /* Custom Info & Alert Boxes */
    .insight-card {
        background-color: #1E293B;
        color: #F8FAFC !important;
        border-left: 5px solid #3B82F6;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .insight-card b, .insight-card strong {
        color: #60A5FA !important;
    }
    .prediction-box-churn {
        background-color: #450A0A;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        color: #FCA5A5 !important;
    }
    .prediction-box-stay {
        background-color: #052E16;
        border: 2px solid #22C55E;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        color: #86EFAC !important;
    }
</style>
""", unsafe_allow_html=True)

# Cached Data & ML Model Handlers
@st.cache_data
def get_cleaned_data():
    raw_df = load_raw_dataset()
    clean_df = clean_dataset(raw_df)
    return raw_df, clean_df

@st.cache_resource
def get_ml_results(_clean_df):
    return train_and_evaluate_models(_clean_df)

raw_df, clean_df = get_cleaned_data()
ml_artifacts = get_ml_results(clean_df)

# Sidebar Quick Information
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=70)
st.sidebar.title("Telco Retention Engine")
st.sidebar.caption("Machine Learning Microproject")
st.sidebar.divider()
st.sidebar.info("""
💡 **Quick Info**:
- **Dataset**: IBM Telco Customer Churn (7,043 rows)
- **Algorithms**: Logistic Regression, Random Forest
- **Pipeline**: Anti-Data-Leakage ColumnTransformer
""")

# Top Dashboard Header
st.title("Customer Churn Analysis & Prediction")
st.caption("Machine Learning Based Customer Retention Analytics Dashboard")

# Top Horizontal Navigation Navbar
tab_home, tab_explorer, tab_eda, tab_insights, tab_ml, tab_predict, tab_explain, tab_about = st.tabs([
    "🏠 Home",
    "📊 Dataset Explorer",
    "📈 EDA",
    "💡 Churn Insights",
    "🤖 ML Models",
    "🔮 Predict Churn",
    "🔍 Feature Importance",
    "ℹ️ About Project"
])

# ==============================================================================
# TAB 1: HOME / OVERVIEW
# ==============================================================================
with tab_home:
    st.subheader("📌 Executive Dashboard Overview")

    # KPI Metrics Calculation
    total_cust = len(clean_df)
    churn_cust = (clean_df['Churn_Numeric'] == 1).sum()
    churn_rate = (churn_cust / total_cust * 100) if total_cust > 0 else 0
    avg_charges = clean_df['MonthlyCharges'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Customers</h4>
            <div class="value">{total_cust:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Churned Customers</h4>
            <div class="value" style="color: #EF4444 !important;">{churn_cust:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Churn Rate</h4>
            <div class="value" style="color: #F59E0B !important;">{churn_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Avg Monthly Charges</h4>
            <div class="value" style="color: #10B981 !important;">${avg_charges:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📌 Project Purpose")
    st.write("""
    Customer churn refers to customers discontinuing a company's products or services. In competitive industries like Telecommunications, 
    retaining existing customers is significantly more cost-effective than acquiring new ones. This machine learning application analyzes customer behavior, 
    identifies key churn factors, and provides predictive insights to prevent customer attrition.
    """)

    st.subheader("🔄 End-to-End Analytics & Machine Learning Pipeline")
    st.markdown("""
    <div class="workflow-container">
        <div class="workflow-step">1. Customer Data</div>
        <div class="workflow-arrow">➔</div>
        <div class="workflow-step">2. Data Cleaning</div>
        <div class="workflow-arrow">➔</div>
        <div class="workflow-step">3. Exploratory Analysis</div>
        <div class="workflow-arrow">➔</div>
        <div class="workflow-step">4. Feature Engineering</div>
        <div class="workflow-arrow">➔</div>
        <div class="workflow-step">5. ML Model</div>
        <div class="workflow-arrow">➔</div>
        <div class="workflow-step">6. Churn Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 **Quick Navigation Guide**: Use the top horizontal tabs to explore dataset analytics, benchmark ML classification algorithms, or predict individual customer churn risk.")

# ==============================================================================
# TAB 2: DATASET EXPLORER
# ==============================================================================
with tab_explorer:
    st.subheader("📊 Dataset Explorer & Schema Inspection")

    col1, col2 = st.columns(2)
    col1.metric("Total Rows (Records)", clean_df.shape[0])
    col2.metric("Total Columns (Features)", clean_df.shape[1])

    st.divider()
    st.subheader("Interactive Dataset Viewer")
    num_rows = st.slider("Select number of rows to preview:", min_value=5, max_value=100, value=10, step=5)
    
    # Filter by Churn status
    filter_churn = st.selectbox("Filter dataset by Churn status:", ["All", "Yes (Churned)", "No (Retained)"])
    if filter_churn == "Yes (Churned)":
        display_df = raw_df[raw_df['Churn'] == 'Yes']
    elif filter_churn == "No (Retained)":
        display_df = raw_df[raw_df['Churn'] == 'No']
    else:
        display_df = raw_df

    st.dataframe(display_df.head(num_rows), use_container_width=True)

    t1, t2, t3 = st.tabs(["📋 Data Schema & Types", "🔍 Missing Values Analysis", "📈 Statistical Summary"])
    
    with t1:
        st.write("Column names and data types in the cleaned dataset:")
        schema_df = pd.DataFrame({
            'Column Name': clean_df.columns,
            'Data Type': clean_df.dtypes.astype(str),
            'Non-Null Count': clean_df.notnull().sum().values
        })
        st.dataframe(schema_df, use_container_width=True)
        
    with t2:
        missing_count = clean_df.isnull().sum()
        missing_df = pd.DataFrame({'Column Name': missing_count.index, 'Missing Values': missing_count.values})
        st.write("Missing values summary (Cleaned automatically during preprocessing):")
        st.dataframe(missing_df[missing_df['Missing Values'] > 0] if (missing_df['Missing Values'] > 0).any() else missing_df.head(10), use_container_width=True)
        st.success("✅ Cleaned dataset contains 0 missing values for ML modeling.")
        
    with t3:
        st.write("Descriptive numerical statistics:")
        st.dataframe(clean_df.describe().T, use_container_width=True)

# ==============================================================================
# TAB 3: EXPLORATORY DATA ANALYSIS
# ==============================================================================
with tab_eda:
    st.subheader("📈 Exploratory Data Analysis (10 Visualizations)")

    # Section A & B
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_churn_distribution(clean_df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_churn_by_gender(clean_df), use_container_width=True)

    st.divider()
    # Section C & D
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_churn_by_contract(clean_df), use_container_width=True)
    with c4:
        st.plotly_chart(plot_churn_by_internet(clean_df), use_container_width=True)

    st.divider()
    # Section E & F
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(plot_churn_by_payment(clean_df), use_container_width=True)
    with c6:
        st.plotly_chart(plot_churn_by_tenure(clean_df), use_container_width=True)

    st.divider()
    # Section G & H
    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(plot_monthly_charges_vs_churn(clean_df), use_container_width=True)
    with c8:
        st.plotly_chart(plot_total_charges_vs_churn(clean_df), use_container_width=True)

    st.divider()
    # Section I & J
    c9, c10 = st.columns(2)
    with c9:
        st.plotly_chart(plot_churn_by_senior(clean_df), use_container_width=True)
    with c10:
        st.plotly_chart(plot_churn_by_tech_support(clean_df), use_container_width=True)

# ==============================================================================
# TAB 4: CUSTOMER CHURN INSIGHTS
# ==============================================================================
with tab_insights:
    st.subheader("💡 Dynamically Calculated Customer Insights")

    insights = generate_dynamic_insights(clean_df)

    st.write("Key Empirical Findings:")
    for obs in insights['observations']:
        st.markdown(f'<div class="insight-card">{obs}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Summary Table of High-Risk Indicators")
    
    summary_data = [
        {"Factor": "Contract Type", "Highest Risk Category": insights['top_contract'], "Churn Rate": f"{insights['top_contract_rate']}%"},
        {"Factor": "Internet Service", "Highest Risk Category": insights['top_internet'], "Churn Rate": f"{insights['top_internet_rate']}%"},
        {"Factor": "Payment Method", "Highest Risk Category": insights['top_pmt'], "Churn Rate": f"{insights['top_pmt_rate']}%"},
        {"Factor": "Tech Support", "Highest Risk Category": "No Tech Support", "Churn Rate": f"{insights['tech_no_rate']}%"},
        {"Factor": "Tenure", "Highest Risk Category": "Short Tenure (< 12 mos)", "Average Months": f"{insights['avg_tenure_churned']} months"},
        {"Factor": "Monthly Charges", "Highest Risk Category": "High Monthly Charges", "Average Charges": f"${insights['avg_monthly_churned']}"}
    ]
    st.table(pd.DataFrame(summary_data))

# ==============================================================================
# TAB 5: MACHINE LEARNING MODEL
# ==============================================================================
with tab_ml:
    st.subheader("🤖 Machine Learning Model Benchmarks")

    models = ml_artifacts['models']
    best_model_name = ml_artifacts['best_model_name']

    st.success(f"🏆 **Best Performing Model:** **{best_model_name}** (Selected based on highest F1 Score)")

    # Model Metrics Comparison Table
    st.subheader("📊 Model Performance Comparison")
    
    comparison_rows = []
    for name, data in models.items():
        comparison_rows.append({
            'Model': name,
            'Accuracy': f"{data['accuracy']:.4f}",
            'Precision': f"{data['precision']:.4f}",
            'Recall': f"{data['recall']:.4f}",
            'F1 Score': f"{data['f1_score']:.4f}"
        })
        
    comp_df = pd.DataFrame(comparison_rows)
    st.dataframe(comp_df, use_container_width=True)

    st.subheader("🧩 Confusion Matrix Comparison")
    col1, col2 = st.columns(2)
    
    for idx, (name, data) in enumerate(models.items()):
        cm = data['confusion_matrix']
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
            x=['Retained (0)', 'Churned (1)'],
            y=['Retained (0)', 'Churned (1)'],
            title=f"<b>Confusion Matrix: {name}</b>",
            color_continuous_scale="Blues",
            template="plotly_dark"
        )
        if idx == 0:
            col1.plotly_chart(fig_cm, use_container_width=True)
        else:
            col2.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("ℹ️ ML Pipeline & Anti-Data-Leakage Architecture"):
        st.markdown("""
        - **Stratified Split:** 80% Training Data, 20% Testing Data preserving target balance (`stratify=y`).
        - **ColumnTransformer Pipeline:** 
          - Numeric Features (`tenure`, `MonthlyCharges`, `TotalCharges`) normalized using `StandardScaler`.
          - Categorical Features transformed using `OneHotEncoder(handle_unknown='ignore')`.
        - **Data Leakage Guarantee:** All scaler parameters and categorical dictionaries are fitted strictly on the **Training Set** inside Scikit-learn `Pipeline` objects.
        """)

# ==============================================================================
# TAB 6: CHURN PREDICTION
# ==============================================================================
with tab_predict:
    st.subheader("🔮 Real-Time Customer Churn Predictor")

    best_pipeline = ml_artifacts['models'][ml_artifacts['best_model_name']]['pipeline']

    st.write("Enter Customer Profile Attributes:")
    
    with st.form(key="prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12, step=1)
            
        with col2:
            st.markdown("#### Subscribed Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            
        with col3:
            st.markdown("#### Contract & Billing")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=70.0, step=0.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=round(monthly_charges * tenure, 2), step=10.0)
            
        submit_btn = st.form_submit_button("🚀 PREDICT CHURN RISK", use_container_width=True)

    if submit_btn:
        customer_dict = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        result = predict_single_customer(best_pipeline, customer_dict)
        
        st.divider()
        st.subheader("🎯 Prediction Result & Risk Assessment")
        
        if result['is_churn']:
            st.markdown(f"""
            <div class="prediction-box-churn">
                <h2 style="color: #EF4444; margin-bottom: 5px;">⚠️ {result['status_label']}</h2>
                <h3 style="color: #FCA5A5;">Estimated Churn Risk: {result['churn_probability']}%</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box-stay">
                <h2 style="color: #22C55E; margin-bottom: 5px;">✅ {result['status_label']}</h2>
                <h3 style="color: #86EFAC;">Estimated Churn Risk: {result['churn_probability']}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.write("Customer Churn Probability Gauge:")
        st.progress(float(result['churn_probability']) / 100.0)
        
        st.subheader("💡 Recommended Actionable Retention Strategies")
        for rec in result['recommendations']:
            st.markdown(f'<div class="insight-card">{rec}</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 7: MODEL EXPLAINABILITY
# ==============================================================================
with tab_explain:
    st.subheader("🔍 Model Explainability & Feature Importance")

    fi_df = ml_artifacts['feature_importances']

    fig_fi = px.bar(
        fi_df.sort_values(by='Importance', ascending=True),
        x='Importance',
        y='CleanFeature',
        orientation='h',
        title="<b>Top 10 Most Influential Features for Churn Prediction</b>",
        labels={'Importance': 'Gini Feature Importance Weight', 'CleanFeature': 'Customer Feature'},
        color='Importance',
        color_continuous_scale="Blues",
        template="plotly_dark"
    )
    fig_fi.update_layout(height=480, showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("💡 Simple Explanation for Viva")
    st.markdown("""
    - **Feature Importance Definition:** Feature importance measures how much each input variable contributes to reducing uncertainty (impurity) across decision trees in the Random Forest ensemble.
    - **Top Driving Variables:** 
      1. **Tenure & Monthly Charges:** Higher tenure strongly correlates with retention, while higher monthly charges increase churn likelihood.
      2. **Contract Type (Month-to-month):** Customers on flexible short-term contracts have higher freedom to switch providers.
      3. **Internet Service (Fiber Optic) & Tech Support:** Technical service quality and support access heavily impact customer satisfaction.
    """)

# ==============================================================================
# TAB 8: ABOUT PROJECT
# ==============================================================================
with tab_about:
    st.subheader("ℹ️ About This Microproject")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Project Profile")
        st.markdown("""
        - **Project Title:** Customer Churn Analysis and Prediction
        - **Category:** Machine Learning & Web Analytics Microproject
        - **Primary Tech Stack:** Python 3.10+, Streamlit, Pandas, NumPy, Scikit-Learn, Plotly
        - **Dataset:** IBM Telco Customer Churn Dataset (7,043 rows, 21 attributes)
        """)
        
    with col2:
        st.subheader("⚙️ Key Engineering Highlights")
        st.markdown("""
        - **End-to-End Pipeline:** Data Loading ➔ Data Cleaning ➔ Exploratory Analysis ➔ ML Pipelines ➔ Real-Time GUI Predictions.
        - **Data Leakage Safeguard:** Utilizes Scikit-learn `Pipeline` and `ColumnTransformer` to fit scalers strictly on training folds.
        - **Responsive Dashboard:** Built with modern CSS cards and interactive Plotly visualization components.
        """)

    st.divider()
    st.subheader("📚 Viva Defense Sample Questions & Answers")
    
    with st.expander("Q1: What is Customer Churn and why is ML useful for predicting it?"):
        st.write("Answer: Customer churn is the rate at which customers discontinue doing business with a firm. ML models identify complex non-linear patterns in customer behavior (like contract type, monthly charges, and tech support usage) to flag high-risk customers proactively before they leave.")

    with st.expander("Q2: How did you prevent Data Leakage during feature scaling and encoding?"):
        st.write("Answer: We wrapped feature scaling (`StandardScaler`) and encoding (`OneHotEncoder`) inside a Scikit-Learn `Pipeline` with `ColumnTransformer`. Preprocessing parameters were fitted strictly on `X_train` during `pipeline.fit()`, preventing test data distributions from leaking into model training.")

    with st.expander("Q3: Why compare Logistic Regression and Random Forest?"):
        st.write("Answer: Logistic Regression serves as a linear baseline model that is highly interpretable. Random Forest is a non-linear ensemble tree algorithm that captures complex feature interactions. Comparing both ensures we select the optimal model based on evaluation metrics like F1-score.")
