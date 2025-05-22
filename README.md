# Anomaly Detection in Time-Series Transaction Data

This project involves analyzing a time-series transaction dataset to **detect anomalies** using statistical techniques, feature engineering, and a supervised machine learning model (Random Forest). The goal is to identify abnormal transactions and understand their distribution across time and agencies.

## Dataset

- **File**: `anomaly_detection.csv`
- **Columns**:
  - `date`: Timestamp of the transaction
  - `value`: Transaction value
  - `anomaly`: Binary indicator (1 = anomaly, 0 = normal)
  - `agency`: Categorical column representing the source agency

## Project Overview

### 1. Data Preprocessing
- Converted `date` to datetime format.
- Visualized class imbalance using a pie chart.
- Displayed anomaly distribution by `agency`.

### 2. Visualization
- Line plot showing normal vs anomaly transactions over time.
- Histogram comparing value distributions for anomalies vs normal.
- Weekly anomaly trends.
- Seasonal analysis (quarterly anomaly distribution with anomaly rates).

### 3. Feature Engineering
- Date-based features:
  - `day_of_week`, `month`, `is_weekend`, `quarter`
- Rolling statistics:
  - `value_rolling_mean_7`: 7-day rolling mean
  - `value_rolling_std_7`: 7-day rolling standard deviation
  - `z_score`: calculated using rolling mean and std
- Lag features:
  - `value_lag_1`: previous day’s transaction value
- Categorical encoding:
  - One-hot encoding for `agency`, `day_of_week`, and `quarter`

### 4. Exploratory Analysis
- Scatterplot of values colored by agency
- Sample agency trends showing:
  - Raw value
  - 7-day rolling average
  - Anomalies highlighted

### 5. Modeling
- Scaled features using `MinMaxScaler`
- Trained a `Random Forest Classifier`
- Evaluation Metrics:
  - Confusion Matrix
  - Accuracy Score
  - Classification Report
- Feature Importance:
  - Extracted and plotted top 10 features contributing to anomaly detection

## Visualizations Included

- Pie Chart: Proportion of anomalies
- Line Plot: Anomalies over time
- Histogram: Value distribution (normal vs anomaly)
- Quarterly Anomalies: Bar and line plots
- Day-of-week anomaly count: Countplot
- Feature Importance: Bar chart
- Time-Series Trend by Agency: Scatterplot

## Model Performance

- Algorithm: `RandomForestClassifier`
- Evaluation:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Top 10 most important features visualized

## Key Insights

- Anomalies are not uniformly distributed across agencies.
- Weekly and quarterly seasonal patterns are clearly visible.
- Rolling statistics and lag features significantly boost model performance.
- Categorical features like agency and day_of_week are crucial predictors.

## Requirements

Install all required Python libraries using:

```bash
pip install pandas matplotlib seaborn scikit-learn
|- anomaly_detection.csv
|- anomaly_detection.ipynb / anomaly_detection.py
|- README.md


## git clone https://github.com/Bhavya1663-thedatasage/Anomaly-Detection.git

> © 2025 Shiva Bhavya Sree Muttireddy. All rights reserved.  
> This project is part of a course submission and is shared only for academic purposes.  
> Reproduction, reuse, or modification of any content is not permitted.

