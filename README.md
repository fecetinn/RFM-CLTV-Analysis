# 📊 Advanced Customer Analytics: RFM & CLTV Analysis with BG-NBD and Gamma-Gamma Models

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Analytics](https://img.shields.io/badge/Analytics-RFM%20%7C%20CLTV%20%7C%20BG--NBD%20%7C%20Gamma--Gamma-brightgreen)
![Models](https://img.shields.io/badge/Models-Probabilistic%20%7C%20Statistical-orange)
![Status](https://img.shields.io/badge/Status-Complete-success)

> **Comprehensive customer analytics framework combining RFM segmentation with probabilistic CLTV prediction using BG-NBD and Gamma-Gamma models for data-driven customer value optimization and strategic marketing decisions.**

---

## 🌟 Overview

This project implements an advanced customer analytics pipeline that combines traditional RFM (Recency, Frequency, Monetary) segmentation with sophisticated probabilistic models (BG-NBD and Gamma-Gamma) to predict Customer Lifetime Value (CLTV). The framework provides actionable insights for customer segmentation, retention strategies, and marketing resource allocation through comprehensive analysis and visualization tools.

### 🎯 Key Features

- **Automated Data Preprocessing Pipeline** with outlier detection and feature engineering
- **Comprehensive RFM Analysis** with 10 distinct customer segments
- **Probabilistic CLTV Modeling** using BG-NBD and Gamma-Gamma models
- **Multi-period CLTV Predictions** (1 week to 1 year horizons)
- **Advanced Customer Ranking Analysis** with temporal comparisons
- **Rich Console Reporting** with tabulated outputs
- **Interactive Visualizations** for business insights

---

## 🗂 Table of Contents

- [🌟 Overview](#-overview)
- [📊 Dataset Description](#-dataset-description)
- [🎯 Business Problem](#-business-problem)
- [🛠 Methodology Pipeline](#-methodology-pipeline)
  - [Data Preprocessing](#data-preprocessing)
  - [RFM Analysis](#rfm-analysis)
  - [CLTV Modeling](#cltv-modeling)
  - [Customer Segmentation](#customer-segmentation)
- [📈 Model Results](#-model-results)
- [💡 Key Insights](#-key-insights)
- [🎯 Business Strategy Recommendations](#-business-strategy-recommendations)
- [🚀 Quick Start](#-quick-start)
- [📂 Project Structure](#-project-structure)
- [🔮 Future Enhancements](#-future-enhancements)
- [🛠 Tech Stack](#-tech-stack)
- [📄 License](#-license)
- [📫 Contact](#-contact)

---

## 📊 Dataset Description

### Online Retail II Dataset

The analysis uses the **Online Retail II** dataset, containing transactional data from a UK-based online retail company specializing in unique all-occasion gifts.

**Dataset Structure:** 
- **File Format**: Excel with two sheets
- **Sheet 1**: "Year 2009-2010" - Transactions from 2009-2010
- **Sheet 2**: "Year 2010-2011" - Transactions from 2010-2011
- **Combined Data**: ~500K transactions after merging both sheets

| Variable | Description | Type | Processing Notes |
|----------|-------------|------|------------------|
| `Invoice` | Invoice number (unique per transaction) | String | Cancelled orders (starting with 'C') are filtered |
| `StockCode` | Product code | String | Used for product identification |
| `Description` | Product name | String | Product description |
| `Quantity` | Number of products per transaction | Integer | Used to calculate total price |
| `InvoiceDate` | Invoice date and time | Datetime | Converted to datetime format |
| `Price` | Unit price | Float | Multiplied by quantity for total value |
| `Customer ID` | Unique customer identifier | Integer | Primary key for customer analysis |
| `Country` | Customer's country | String | Geographic segmentation possible |

### Data Quality & Preprocessing
- **Missing Values**: Removed using `dropna()`
- **Cancelled Orders**: Filtered out (invoices containing 'C')
- **Feature Engineering**: `TotalPrice = Quantity × Price`
- **Outlier Treatment**: IQR method with customizable thresholds
- **Date Processing**: All date columns converted to datetime64[ns]

> *Note: Download `online_retail_II.xlsx` from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) and place it in the project root directory.*

---

## 🎯 Business Problem

**Challenge**: How can we identify and quantify the future value of existing customers to optimize marketing spend and improve retention strategies?

### Key Business Questions
1. **Customer Segmentation**: Which customers belong to which behavioral segments?
2. **Value Prediction**: What is the expected lifetime value of each customer?
3. **Resource Allocation**: How should marketing budgets be distributed across segments?
4. **Retention Priority**: Which customers require immediate retention efforts?
5. **Growth Opportunities**: Which segments have the highest growth potential?

---

## 🛠 Methodology Pipeline

### Data Preprocessing

```python
# Key preprocessing steps
1. Load and combine data from multiple sheets
2. Remove missing values and cancelled orders
3. Calculate TotalPrice = Quantity × Price
4. Apply outlier detection using IQR method
5. Convert date columns to datetime format
```

Additional preprocessing utilities include:
- `comprehensive_date_detection()`: Intelligent date column identification
- `grab_col_names()`: Automated column classification (numerical, categorical, cardinal, date)
- `outlier_threshold_IQR()`: Outlier detection using IQR method
- `replace_with_threshold()`: Outlier replacement with threshold values
- `check_df_tabulate()`: Comprehensive DataFrame analysis and reporting

### RFM Analysis

The RFM methodology segments customers based on three key metrics:

| Metric | Description | Scoring |
|--------|-------------|---------|
| **Recency** | Days since last purchase | 1-5 (5 = most recent) |
| **Frequency** | Total number of purchases | 1-5 (5 = most frequent) |
| **Monetary** | Total spending amount | 1-5 (5 = highest value) |

**Segmentation Map:**
```python
{
    'Champions': '[4-5][4-5]',
    'Loyal Customers': '[3-4][4-5]',
    'Potential Loyalists': '[4-5][2-3]',
    'New Customers': '51',
    'Promising': '41',
    'Need Attention': '33',
    'About to Sleep': '3[1-2]',
    'At Risk': '[1-2][3-4]',
    'Can't Lose': '[1-2]5',
    'Hibernating': '[1-2][1-2]'
}
```

### CLTV Modeling

#### BG-NBD Model (Beta Geometric/Negative Binomial Distribution)
Predicts customer purchase frequency by modeling:
- **Purchase Process**: Transaction frequency (Poisson process)
- **Dropout Process**: Customer churn probability (Geometric distribution)

```python
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(frequency, recency_weekly, tenure_weekly)
```

#### Gamma-Gamma Model
Estimates expected customer monetary value:
- Assumes transaction values vary around customer's average
- Models average transaction values across customers

```python
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(frequency, monetary_average)
```

### Customer Segmentation

CLTV-based segmentation using percentile thresholds:

| Segment | Percentile Range | Business Value |
|---------|------------------|----------------|
| **A** | 90-100% | Premium customers |
| **B** | 75-90% | High-value customers |
| **C** | 50-75% | Medium-value customers |
| **D** | 25-50% | Low-medium value |
| **E** | 10-25% | Low-value customers |
| **F** | 0-10% | Minimal value |

---

## 📈 Model Results

### RFM Segment Distribution

The `analyze_rfm_segments()` function provides comprehensive segment analysis with:

- **Segment Distribution**: Customer count and percentage per segment
- **Average Metrics**: Mean recency, frequency, and monetary values
- **Business Priority**: Segments sorted by business value hierarchy
- **Key Insights**: Automatic identification of largest, highest value, and most recent segments
  

| Segment | Customer Count | Avg Recency | Avg Frequency | Avg Monetary | % of Total |
|---------|----------------|-------------|---------------|--------------|------------|
| Champions | 830 | 9.5 days | 19.4 | $10959.1 | 14.12% |
| Loyal Customers | 1167 | 67.7 days | 9.8 | $4213.52 | 19.85% |
| Potential Loyalists | 711 | 26.6 days | 2.6 | $1158.35 | 12.10% |
| New Customers | 54 | 11.5 days | 1.0 | $360.67 | 0.92% |
| Promising | 113 | 39.6 days | 1.0 | $321.6 | 1.92% |
| Need Attention | 268  | 114 days | 3.1 | $1278.69 | 4.56% |
| About To Sleep | 384  | 107.2 days | 1.4 | $533.93 | 6.53% |
| At Risk | 754 | 373.4 days | 3.9 | $1381.6 | 12.83% |
| Can't Lose | 71 | 332.2 days | 15.9 | $8355.68 | 1.21% |
| Hibernating | 1526 | 459.8 days | 1.2 | $437.84 | 25.96% |

**Segment Hierarchy (Business Value):**
1. **Champions**: Best customers with high value, frequency, and recent activity
2. **Loyal Customers**: Reliable customers with consistent metrics
3. **Potential Loyalists**: Good customers with growth potential
4. **New Customers**: Recent customers requiring nurturing
5. **Promising**: New customers showing positive signals
6. **Need Attention**: Average customers requiring engagement
7. **About to Sleep**: Customers showing declining activity
8. **At Risk**: Important customers at risk of churning
9. **Can't Lose**: High-value customers with declining recency
10. **Hibernating**: Inactive customers with minimal engagement

### CLTV Predictions Summary

| Segment | Avg Frequency | Avg Monetary ($) | Avg 1W CLTV | Avg 1M CLTV | Avg 3M CLTV | Avg 6M CLTV | Avg 1Y CLTV | Customer Count |
|---------|---------------|------------------|-------------|-------------|-------------|-------------|-------------|----------------|
| F       | 2.4           | 196.16           | 0.08        | 0.30        | 0.83        | 1.48        | 2.46        | 588            |
| E       | 1.6           | 342.92           | 0.72        | 2.73        | 7.36        | 13.00       | 21.40       | 882            |
| D       | 3.1           | 322.44           | 21.74       | 82.25       | 221.67      | 392.72      | 649.94      | 1469           |
| C       | 5.3           | 325.50           | 98.04       | 372.12      | 1008.49     | 1794.56     | 2982.48     | 1469           |
| B       | 9.0           | 438.61           | 233.04      | 882.17      | 2386.52     | 4245.61     | 7059.05     | 882            |
| A       | 23.5          | 928.65           | 1000.66     | 3815.89     | 10421.00    | 18666.00    | 31237.50    | 588            |

**Additional Metrics:**
- `Expected_Avg_Profit`: Predicted average transaction value per customer
- `CLTV_[Period]`: Customer lifetime value for each time period
- `CLTV_Segment`: Customer segment (A-F) based on CLTV percentiles

---

## 💡 Key Features & Capabilities

### 🎯 Analysis Features
- **Comprehensive Data Quality Report**: Automated detection of data types, missing values, and outliers
- **Dynamic RFM Segmentation**: 10 distinct customer segments based on behavior patterns
- **Multi-Period CLTV Forecasting**: Predictions from 1 week to 1 year horizons
- **Customer Ranking Analysis**: Track how customer value rankings change over time
- **Segment Migration Tracking**: Monitor customer movement between segments

### 📊 Technical Capabilities
- **Automated Column Classification**: Intelligent detection of numerical, categorical, cardinal, and date columns
- **Outlier Detection & Treatment**: IQR-based outlier handling with customizable thresholds
- **Rich Console Output**: Tabulated reports with descriptive statistics and insights
- **Visualization Suite**: Distribution plots, correlation heatmaps, and model validation charts

### 🔮 Model Outputs
- **Expected Purchase Frequency**: Number of transactions predicted per time period
- **Expected Average Value**: Predicted monetary value per transaction
- **Customer Lifetime Value**: Combined prediction of frequency × monetary value
- **Segment Classifications**: Both RFM-based and CLTV-based customer groupings

---

## 🎯 Business Strategy Recommendations

### Segment-Specific Strategies

#### 👑 Champions & Loyal Customers
- **VIP Program**: Exclusive previews and personalized service
- **Loyalty Rewards**: Points multiplier and tier benefits
- **Retention Focus**: Personal account managers for top-tier customers
- **Investment Priority**: Allocate 40% of retention budget

#### 🚀 Potential Loyalists & Promising
- **Engagement Campaigns**: Product education and usage tips
- **Cross-selling**: Complementary product recommendations
- **Incentive Programs**: Graduated rewards for increased frequency
- **Investment Priority**: Allocate 30% of acquisition budget

#### ⚠️ At Risk & Can't Lose
- **Win-back Campaigns**: Personalized reactivation offers
- **Feedback Collection**: Understand churn reasons through surveys
- **Special Attention**: Direct outreach from customer success team
- **Investment Priority**: Allocate 20% of retention budget

#### 💤 Hibernating & About to Sleep
- **Automated Re-engagement**: Email sequences with progressive discounts
- **Channel Optimization**: Test different communication channels
- **Low-touch Approach**: Cost-effective automated campaigns
- **Investment Priority**: Allocate 10% of retention budget

### Implementation Example
```python
# Identify and export segments for targeted campaigns
champions = rfm_results[rfm_results['Segment'] == 'champions']
at_risk = rfm_results[rfm_results['Segment'] == 'at_risk']

# Create high-value customer list (top 20% by CLTV)
high_value = cltv_results[cltv_results['CLTV_Segment'].isin(['A', 'B'])]

# Export for marketing automation
champions['Customer ID'].to_csv('vip_customers.csv', index=False)
at_risk['Customer ID'].to_csv('retention_campaign.csv', index=False)
```

---

## 🔮 Future Enhancements

### 📊 Advanced Analytics
- **Machine Learning Models**: XGBoost/CatBoost for CLTV prediction
- **Deep Learning**: LSTM for sequential purchase behavior modeling
- **Survival Analysis**: Cox regression for churn prediction
- **Market Basket Analysis**: Association rules for product recommendations

### 🚀 Technical Infrastructure
- **Real-time Pipeline**: Apache Kafka for streaming analytics
- **Cloud Deployment**: AWS/GCP implementation
- **API Development**: FastAPI for model serving
- **MLOps**: MLflow for experiment tracking

### 📈 Business Intelligence
- **Interactive Dashboard**: Streamlit/Dash application
- **Automated Reporting**: Scheduled email reports
- **A/B Testing Framework**: Experiment tracking system
- **ROI Calculator**: Marketing spend optimization tool

---

## 🛠 Tech Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Lifetimes**: BG-NBD and Gamma-Gamma implementations
- **Matplotlib & Seaborn**: Data visualization
- **Tabulate**: Console table formatting

### Statistical Models
- **BG-NBD Model**: Customer transaction frequency prediction
- **Gamma-Gamma Model**: Customer monetary value estimation
- **RFM Segmentation**: Rule-based customer classification

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📫 Contact

**Fatih Eren Çetin**

<p align="left">
  <a href="https://www.linkedin.com/in/fatih-eren-cetin" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" height="30" />
  </a>
  
  <a href="https://medium.com/@fecetinn" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white" alt="Medium" height="30" />
  </a>
  
  <a href="https://www.kaggle.com/fatiherencetin" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle" height="30" />
  </a>
  
  <a href="https://github.com/fecetinn" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" height="30" />
  </a>

  <a href="https://www.hackerrank.com/profile/fecetinn" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/HackerRank-2EC866?style=for-the-badge&logo=hackerrank&logoColor=white" alt="HackerRank" height="30" />
  </a>
</p>

**Email**: [fatih.e.cetin@gmail.com](mailto:fatih.e.cetin@gmail.com)

---

### 🙏 Acknowledgments

This project demonstrates advanced customer analytics techniques combining traditional RFM analysis with modern probabilistic modeling approaches. Special thanks to the data science community for continuous inspiration and knowledge sharing.

**References:**
- Fader, P. S., Hardie, B. G., & Lee, K. L. (2005). "Counting your customers" the easy way
- Fader, P. S., & Hardie, B. G. (2013). The Gamma-Gamma model of monetary value
- Online Retail II Dataset - UCI Machine Learning Repository

---

*⭐ If you find this project useful, please consider giving it a star!*
