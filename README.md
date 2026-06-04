# 🚀 Startup Success Predictor

An AI-powered web application that predicts startup success and forecasts future revenue using machine learning. Built with Streamlit, scikit-learn, and Plotly.

![Python Version](https://img.shields.io/badge/python-blue.svg)
![Streamlit Version](https://img.shields.io/badge/streamlit-red.svg)
![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression%20%7C%20Linear%20Regression-orange)

### 📊 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://startup-predictor-vqzdtqzhfqpdsus4htue5u.streamlit.app/)

## 📸 Screenshots Gallery

### Dashboard Overview
*Real-time KPIs and success rate visualizations*

<img src="https://github.com/user-attachments/assets/7a6a5de5-bf5e-4355-99c8-e4fe905f23d4" width="800" alt="Dashboard">

*Dashboard showing key metrics, success rate by industry, and funding stage distribution*

### Model Performance Metrics
*Evaluate model accuracy and performance*

<img src="https://github.com/user-attachments/assets/cb76bdfe-5562-4a05-919f-cf3c262275aa" width="800" alt="Model Metrics">

*Classification and regression metrics with confusion matrix*

### Success Prediction

*Predict if your startup will succeed or fail*

<img src="<img width="2226" height="1120" alt="image" src="https://github.com/user-attachments/assets/24b10777-ef8a-4c1d-b190-b3be9d61cacb" />
" width="800" alt="Success Prediction - Success">

*Example of a startup predicted to succeed with 63.8% confidence*

<img src="https://github.com/user-attachments/assets/e1699ccd-0ba0-4763-a4a9-c77326092d02" width="800" alt="Success Prediction - Fail">

*Example of a startup predicted to fail with 76.5% confidence*

### Revenue Forecast
*Forecast future revenue with percentile comparisons*

<img src="https://github.com/user-attachments/assets/5a4e18e5-4a20-40ea-ae79-bef32fbca2b5" width="800" alt="Revenue Forecast">

*Revenue prediction showing $10.2M projected revenue*

### Data Explorer
*Browse and filter through all startup records*

<img src="https://github.com/user-attachments/assets/4a329f3e-ad11-4337-a84f-d68193d7ee7a" width="800" alt="Data Explorer - Success">

*Filtered view showing successful BioTech startups*

<img src="https://github.com/user-attachments/assets/baa96e59-5f24-47b8-932a-ef69e5b477df" width="800" alt="Data Explorer - Fail">

*Filtered view showing failed CleanTech startups*
### Success Probability Gauge
*Visual probability indicator for predictions*


> **Note**: Screenshots are stored in the `screenshots/` folder. Create this folder and add your images with the names mentioned above.

## ✨ Features

### 🏠 Dashboard
- Real-time KPIs (Total Startups, Success Rate, Average Revenue)
- Success rate visualization by industry
- Interactive charts with Plotly
- Funding stage distribution analysis

### 🎯 Success Prediction (Logistic Regression)
- Predict if a startup will succeed or fail
- Confidence scores and probability gauges
- Input parameters:
  - Funding amount ($M)
  - Team size
  - Years active
  - Market size ($M)
  - Number of competitors
  - Product score (1-10)
  - Customer count
  - Marketing spend ($M)
  - Industry & Funding stage

### 💰 Revenue Forecast (Linear Regression)
- Predict future revenue in millions
- Percentile comparison with dataset
- Key input summary
- Performance metrics:
  - R² Score: 0.875
  - RMSE: $1.5M
  - MAPE: ~15%

### 📊 Model Performance Metrics
- Accuracy: 73.4%
- Precision: 71.9%
- Recall: 70.7%
- F1 Score: 0.713
- Confusion matrix visualization
- Classification & Regression metrics

### 📁 Data Explorer
- Browse all 620+ startup records
- Filter by outcome, industry, and funding stage
- Sortable columns with monetary values in $M

## 🔍 Exploratory Data Analysis (EDA)

Before building the models, we performed comprehensive EDA to understand the data:

### Data Overview
- **Dataset Size**: 620 startup records
- **Features**: 14 variables (10 numerical, 4 categorical)
- **Target Variables**: 
  - `success` (Binary classification)
  - `future_revenue_m` (Regression)

### Key Insights Discovered

#### 1. Success Rate Distribution
- Overall success rate: ~50% (balanced dataset)
- Best performing industries: CleanTech (62%), FinTech (58%)
- Industries needing improvement: BioTech (41%), EdTech (43%)

#### 2. Funding Stage Analysis
- Series A startups have highest success rate (68%)
- Pre-Seed stage shows most variance in outcomes
- Series B+ startups show more stable revenue patterns

#### 3. Correlation Analysis
- **Strong positive correlations**:
  - Funding amount → Future revenue (r = 0.72)
  - Team size → Customer count (r = 0.65)
  - Product score → Success rate (r = 0.58)
  
- **Negative correlations**:
  - Competitors → Success rate (r = -0.34)
  - Burn rate → Profitability (r = -0.41)

#### 4. Revenue Patterns
- Average future revenue: $8.2M
- Revenue follows log-normal distribution
- Top 10% of startups generate 40% of total revenue

#### 5. Feature Importance (Initial)
- Most predictive features for success:
  1. Product score (23% importance)
  2. Funding amount (18% importance)
  3. Team size (15% importance)
  4. Market size (12% importance)

## 🛠️ Feature Engineering

We created and transformed features to improve model performance:

### 1. Feature Encoding

#### Label Encoding (Categorical Features)
```python
# Converted categorical variables to numerical
le_industry = LabelEncoder()
le_stage = LabelEncoder()

df["industry_enc"] = le_industry.fit_transform(df["industry"])
df["funding_stage_enc"] = le_stage.fit_transform(df["funding_stage"])
