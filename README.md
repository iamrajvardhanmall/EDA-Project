Bank Customer Churn Analysis - Exploratory Data Analysis (EDA)
📌 Overview
This project performs an in-depth Exploratory Data Analysis (EDA) on bank customer churn data to identify factors influencing customer attrition. The analysis includes statistical tests, visualizations, and an interactive dashboard to help banks understand and mitigate customer churn.

📊 Dataset
The dataset (Bank_Churn.csv) contains customer information with the following key features:

Demographics: Geography, Gender, Age

Financial Attributes: CreditScore, Balance, EstimatedSalary

Behavioral Metrics: Tenure, NumOfProducts, IsActiveMember, HasCrCard

Target Variable: Exited (Churn status)

🔍 Key Findings
Demographic Insights:

Older customers (40+) show higher churn rates

Female customers in Germany have the highest churn rate

Geography significantly impacts churn (Germany > France > Spain)

Financial Patterns:

Credit score shows minimal impact on churn

Salary level doesn't correlate with churn

Customers with mid-range balances (50-150k) are most likely to churn

Behavioral Trends:

Inactive members churn at 2.5x the rate of active members

Customers with 2 products have lowest churn rate

Tenure shows no clear relationship with churn

📈 Statistical Significance
Highly Significant Predictors (p < 0.001):

Age, IsActiveMember, NumOfProducts, Balance

Moderate Predictors:

Geography, Gender

Non-Significant Factors:

CreditScore, EstimatedSalary, Tenure

🛠️ Technical Implementation
Libraries Used
Pandas, NumPy (Data manipulation)

Matplotlib, Seaborn, Plotly (Visualization)

SciPy, StatsModels (Statistical testing)

IPython Widgets (Interactive dashboard)

Key Analyses Performed
Multicollinearity check using VIF

Distribution analysis of all features

Correlation matrix heatmap

Chi-square tests for categorical variables

Z-tests for numerical variables

Interactive dashboard with filtering capabilities
