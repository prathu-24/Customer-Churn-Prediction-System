import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Styling constants
COLOR_RETAINED = "#3B82F6"  # Vibrant Royal Blue
COLOR_CHURNED = "#EF4444"   # Coral Red
COLOR_MAP = {"No": COLOR_RETAINED, "Yes": COLOR_CHURNED}

TEMPLATE = "plotly_dark"

def plot_churn_distribution(df):
    churn_counts = df['Churn'].value_counts().reset_index()
    churn_counts.columns = ['Churn', 'Count']
    churn_counts['Percentage'] = (churn_counts['Count'] / churn_counts['Count'].sum() * 100).round(2)
    
    fig = px.bar(
        churn_counts,
        x='Churn',
        y='Count',
        color='Churn',
        color_discrete_map=COLOR_MAP,
        text=churn_counts['Percentage'].apply(lambda x: f"{x}%"),
        title="<b>Overall Customer Churn Distribution</b>",
        labels={'Count': 'Number of Customers', 'Churn': 'Customer Status'},
        template=TEMPLATE
    )
    fig.update_traces(textposition='outside', textfont_size=14)
    fig.update_layout(showlegend=False, yaxis_range=[0, max(churn_counts['Count']) * 1.15], height=400)
    return fig

def plot_churn_by_category(df, column, title, label_map=None):
    df_plot = df.copy()
    if label_map:
        df_plot[column] = df_plot[column].map(label_map)
        
    counts = df_plot.groupby([column, 'Churn']).size().reset_index(name='Count')
    totals = df_plot.groupby(column).size().reset_index(name='Total')
    counts = counts.merge(totals, on=column)
    counts['Percentage'] = (counts['Count'] / counts['Total'] * 100).round(1)
    
    fig = px.bar(
        counts,
        x=column,
        y='Count',
        color='Churn',
        barmode='group',
        color_discrete_map=COLOR_MAP,
        text=counts['Percentage'].apply(lambda x: f"{x}%"),
        title=f"<b>{title}</b>",
        labels={'Count': 'Customer Count', column: column.replace('_', ' ')},
        template=TEMPLATE
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(legend_title_text='Churn Status', height=420)
    return fig

def plot_churn_by_gender(df):
    return plot_churn_by_category(df, 'gender', 'Customer Churn by Gender')

def plot_churn_by_contract(df):
    return plot_churn_by_category(df, 'Contract', 'Customer Churn by Contract Type')

def plot_churn_by_internet(df):
    return plot_churn_by_category(df, 'InternetService', 'Customer Churn by Internet Service Provider')

def plot_churn_by_payment(df):
    return plot_churn_by_category(df, 'PaymentMethod', 'Customer Churn by Payment Method')

def plot_churn_by_senior(df):
    label_map = {0: 'Non-Senior Citizen', 1: 'Senior Citizen'}
    return plot_churn_by_category(df, 'SeniorCitizen', 'Customer Churn by Senior Citizen Status', label_map=label_map)

def plot_churn_by_tech_support(df):
    return plot_churn_by_category(df, 'TechSupport', 'Customer Churn by Tech Support Availability')

def plot_churn_by_tenure(df):
    fig = px.histogram(
        df,
        x='tenure',
        color='Churn',
        barmode='overlay',
        color_discrete_map=COLOR_MAP,
        nbins=30,
        title="<b>Customer Tenure Distribution vs Churn</b>",
        labels={'tenure': 'Tenure (Months)', 'count': 'Number of Customers'},
        template=TEMPLATE,
        opacity=0.75
    )
    fig.update_layout(legend_title_text='Churn Status', height=420)
    return fig

def plot_monthly_charges_vs_churn(df):
    fig = px.box(
        df,
        x='Churn',
        y='MonthlyCharges',
        color='Churn',
        color_discrete_map=COLOR_MAP,
        points="outliers",
        title="<b>Monthly Charges ($) vs Churn Status</b>",
        labels={'MonthlyCharges': 'Monthly Charges ($)', 'Churn': 'Churn Status'},
        template=TEMPLATE
    )
    fig.update_layout(showlegend=False, height=420)
    return fig

def plot_total_charges_vs_churn(df):
    fig = px.box(
        df,
        x='Churn',
        y='TotalCharges',
        color='Churn',
        color_discrete_map=COLOR_MAP,
        points="outliers",
        title="<b>Total Charges ($) vs Churn Status</b>",
        labels={'TotalCharges': 'Total Charges ($)', 'Churn': 'Churn Status'},
        template=TEMPLATE
    )
    fig.update_layout(showlegend=False, height=420)
    return fig
