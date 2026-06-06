# AI-Powered Data Analytics Dashboard

## Overview

An intelligent data analytics platform built using Streamlit and Gemini AI that automatically analyzes datasets, generates insights, and explains visualizations in business-friendly language.

## Features

* Automatic dataset understanding (AI + rule-based schema detection)
* Dynamic dashboards for any CSV dataset
* AI-generated chart explanations
* Automatic business insights generation
* Interactive data exploration
* Smart column detection (works across different datasets)

## Tech Stack

* Python
* Streamlit
* Pandas, NumPy
* Google Gemini API (LLM)
* Matplotlib

## How It Works

1. Upload any CSV dataset
2. AI detects column meanings (revenue, date, category, etc.)
3. Dashboard is generated automatically
4. AI explains charts and generates insights

## Setup Instructions

```bash
git clone https://github.com/YOUR_USERNAME/ai-data-analytics-dashboard.git
cd ai-data-analytics-dashboard
pip install -r requirements.txt
```

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

Run the app:

```bash
streamlit run app.py
```

## Key Highlights

* Works with multiple datasets without hardcoding column names
* Uses AI to generate business insights
* Designed for real-world data analytics use cases

## Future Improvements

* Chat with dataset (AI assistant)
* Advanced anomaly detection
* Predictive analytics
