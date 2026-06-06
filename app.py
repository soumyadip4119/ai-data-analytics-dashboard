import streamlit as st
from google import genai
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# ==============================
# API Setup
# ==============================
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="AI Data Analytics Platform", layout="wide")

st.title("AI-Powered Data Analytics Platform")

# ==============================
# File Upload
# ==============================
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

# ==============================
# Helper Functions
# ==============================
def is_numeric_column(series):
    cleaned = series.astype(str).str.replace(r'[\$,]', '', regex=True)
    converted = pd.to_numeric(cleaned, errors='coerce')
    return converted.notna().mean() > 0.7

def is_date_column(series):
    converted = pd.to_datetime(series, errors='coerce', dayfirst=True)
    return converted.notna().mean() > 0.7

# ==============================
# AI Chart Explanation
# ==============================
def generate_chart_explanation(data, chart_type, context=""):
    try:
        # Clean data
        if isinstance(data, dict):
            clean_data = {
                k: float(v) if isinstance(v, (int, float, np.number)) else 0
                for k, v in data.items()
            }
        else:
            clean_data = str(data)[:1000]

        prompt = f"""
        You are a business analyst.

        Explain the chart in simple business terms.

        Chart Type: {chart_type}
        Context: {context}

        Data Summary:
        {clean_data}

        Keep it:
        - Short (2-3 lines)
        - Insightful
        - Business-friendly
        """

        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        st.warning(f"AI Error: {str(e)}")

        return f"""
        Basic Insight:
        The {chart_type} shows variation in the data. 
        There may be trends or inconsistencies worth further investigation.
        """

# ==============================
# Rule-Based Detection
# ==============================
def detect_columns_rule(df):
    col_map = {}

    for col in df.columns:
        c = col.lower()

        if 'date' in c:
            col_map['date'] = col
        elif any(x in c for x in ['sales', 'revenue', 'amount', 'total']):
            col_map['revenue'] = col
        elif any(x in c for x in ['profit', 'margin']):
            col_map['profit'] = col
        elif any(x in c for x in ['quantity', 'qty', 'units']):
            col_map['quantity'] = col
        elif 'discount' in c:
            col_map['discount'] = col
        elif any(x in c for x in ['shipping', 'delivery']):
            col_map['shipping_cost'] = col

        elif any(x in c for x in ['state', 'region', 'location']):
            col_map['region'] = col
        elif 'city' in c:
            col_map['city'] = col
        elif 'country' in c:
            col_map['country'] = col

        elif 'sub-category' in c or 'subcategory' in c:
            col_map['sub_category'] = col
        elif 'category' in c:
            col_map['category'] = col
        elif any(x in c for x in ['product', 'item']):
            col_map['product'] = col

        elif any(x in c for x in ['customer', 'client']):
            col_map['customer'] = col
        elif 'segment' in c:
            col_map['segment'] = col

    return col_map

# ==============================
# Data-Based Detection
# ==============================
def detect_columns_data(df):
    col_map = {}

    for col in df.columns:
        if is_date_column(df[col]) and 'date' not in col_map:
            col_map['date'] = col
        elif is_numeric_column(df[col]) and 'revenue' not in col_map:
            col_map['revenue'] = col

    return col_map

# ==============================
# AI Schema Detection
# ==============================
def detect_schema_ai(columns, sample):
    prompt = f"""
    You are a senior data analyst.

    Classify dataset columns into roles.

    Return ONLY valid JSON.

    Columns:
    {columns}

    Sample Data:
    {sample}
    """

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        return json.loads(response.text)
    except:
        return {}

# ==============================
# Main App
# ==============================
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df.head())

    rule_map = detect_columns_rule(df)
    data_map = detect_columns_data(df)
    sample = df.head(5).to_dict()

    ai_map = detect_schema_ai(df.columns.tolist(), sample)

    schema = {**ai_map, **data_map, **rule_map}

    st.subheader("Detected Schema")
    st.json(schema)

    def get_index(col_name):
        return df.columns.get_loc(col_name) if col_name in df.columns else 0

    st.sidebar.header("Column Mapping")

    date_col = st.sidebar.selectbox("Date", df.columns, index=get_index(schema.get("date", df.columns[0])))
    revenue_col = st.sidebar.selectbox("Revenue", df.columns, index=get_index(schema.get("revenue", df.columns[0])))
    profit_col = st.sidebar.selectbox("Profit", df.columns, index=get_index(schema.get("profit", df.columns[0])))
    region_col = st.sidebar.selectbox("Region", df.columns, index=get_index(schema.get("region", df.columns[0])))
    category_col = st.sidebar.selectbox("Category", df.columns, index=get_index(schema.get("category", df.columns[0])))
    city_col = st.sidebar.selectbox("City", df.columns, index=get_index(schema.get("city", df.columns[0])))
    segment_col = st.sidebar.selectbox("Segment", df.columns, index=get_index(schema.get("segment", df.columns[0])))
    discount_col = st.sidebar.selectbox("Discount", df.columns, index=get_index(schema.get("discount", df.columns[0])))

    # ==============================
    # Data Cleaning
    # ==============================
    df[revenue_col] = df[revenue_col].astype(str).str.replace(r'[\$,]', '', regex=True)
    df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)

    if discount_col:
        df[discount_col] = pd.to_numeric(df[discount_col], errors='coerce')

    df = df.dropna(subset=[revenue_col])

    tab1, tab2, tab3 = st.tabs(["Overview", "Deep Dive", "AI Insights"])

    # ==============================
    # OVERVIEW
    # ==============================
    with tab1:
        st.subheader("Sales Trend")

        monthly_sales = df.groupby(df[date_col].dt.to_period('M'))[revenue_col].sum()
        st.line_chart(monthly_sales)

        explanation = generate_chart_explanation(
            monthly_sales.describe().to_dict(),
            "Time Series",
            "Monthly revenue trend"
        )
        st.info(explanation)

    # ==============================
    # DEEP DIVE
    # ==============================
    with tab2:

        if category_col:
            st.subheader("Category Analysis")
            cat_sales = df.groupby(category_col)[revenue_col].sum()
            st.bar_chart(cat_sales)

            explanation = generate_chart_explanation(
                cat_sales.describe().to_dict(),
                "Bar Chart",
                "Revenue by category"
            )
            st.info(explanation)

        if discount_col:
            st.subheader("Discount vs Revenue")

            scatter_df = df[[discount_col, revenue_col]].dropna()

            st.scatter_chart(scatter_df)

            # SMART SUMMARY (IMPORTANT FIX)
            sample_data = scatter_df.sample(min(200, len(scatter_df)))

            summary = {
                "correlation": float(sample_data.corr().iloc[0, 1]),
                "discount_mean": float(sample_data[discount_col].mean()),
                "revenue_mean": float(sample_data[revenue_col].mean()),
                "points": len(sample_data)
            }

            explanation = generate_chart_explanation(
                summary,
                "Scatter Plot",
                "Discount vs revenue relationship"
            )

            st.info(explanation)

    # ==============================
    # AI INSIGHTS
    # ==============================
    with tab3:

        st.subheader("Automatic Insights")

        auto_summary = {
            "total_revenue": float(df[revenue_col].sum()),
            "avg_revenue": float(df[revenue_col].mean()),
            "rows": len(df)
        }

        auto_insight = generate_chart_explanation(
            auto_summary,
            "Dataset Overview",
            "Overall business performance"
        )

        st.info(auto_insight)

        question = st.text_input("Ask a business question")

        if st.button("Generate Insights"):

            prompt = f"""
            Analyze this dataset and answer:

            {question}

            Provide:
            - Key insights
            - Risks
            - Recommendations
            """

            try:
                response = client.models.generate_content(
                    model="models/gemini-2.5-flash",
                    contents=prompt
                )
                st.markdown(response.text)
            except Exception as e:
                st.error(f"AI failed: {str(e)}")

else:
    st.info("Upload a CSV file to start.")