Bank Customer Churn Analysis - Exploratory Data Analysis (EDA)<br>
📌 Overview<br>
This project performs an in-depth Exploratory Data Analysis (EDA) on bank customer churn data to identify factors influencing customer attrition. The analysis includes statistical tests, visualizations, and an interactive dashboard to help banks understand and mitigate customer churn.<br>

📊 Dataset<br>
The dataset (Bank_Churn.csv) contains customer information with the following key features:<br>
a.) Demographics: Geography, Gender, Age<br>
b.) Financial Attributes: CreditScore, Balance, EstimatedSalary<br>
c.) Behavioral Metrics: Tenure, NumOfProducts, IsActiveMember, HasCrCard<br>
d.) Target Variable: Exited (Churn status)<br>

🔍 Key Findings<br>
Demographic Insights:<br>
a.) Older customers (40+) show higher churn rates<br>
b.) Female customers in Germany have the highest churn rate<br>
c.) Geography significantly impacts churn (Germany > France > Spain)<br>

Financial Patterns:<br>
a.) Credit score shows minimal impact on churn<br>
b.) Salary level doesn't correlate with churn<br>
c.) Customers with mid-range balances (50-150k) are most likely to churn<br>

Behavioral Trends:<br>
a.) Inactive members churn at 2.5x the rate of active members<br>
b.) Customers with 2 products have lowest churn rate<br>
c.) Tenure shows no clear relationship with churn<br>

🛠️ Technical Implementation<br>
a.) Libraries Used<br>
i.) Pandas, NumPy (Data manipulation)<br>
ii.) Matplotlib, Seaborn, Plotly (Visualization)<br>
iii.) SciPy, StatsModels (Statistical testing)<br>
iv.) IPython Widgets (Interactive dashboard)<br>


Chi-square tests for categorical variables

Z-tests for numerical variables

Interactive dashboard with filtering capabilities
