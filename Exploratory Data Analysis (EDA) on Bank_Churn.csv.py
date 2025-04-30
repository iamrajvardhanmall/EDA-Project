# PROBLEM STATEMENT:-
# The goal of this project is to perform an Exploratory Data Analysis (EDA) on the "Bank_Churn.csv" dataset to understand the factors influencing customer churn in a banking context. By analyzing customer demographics, financial attributes, and behavioral patterns, the project aims to identify key predictors of churn and provide actionable insights for the bank to improve customer retention strategies.

# OBJECTIVES:-
# 1.) Understand Dataset Structure: Explore the dataset's size, features, data types, and presence of missing values to ensure data quality.
# 2.) Analyze Customer Demographics: Examine the distribution of demographic variables (e.g., Geography, Gender, Age) and their relationship with churn.
# 3.) Investigate Financial Attributes: Assess financial features such as Credit Score, Estimated Salary, and Balance to determine their impact on churn.
# 4.) Evaluate Behavioral Patterns: Study customer engagement metrics like IsActiveMember, Tenure, and NumOfProducts to identify behavioral trends associated with churn.
# 5.) Visualize Relationships: Use visualizations (e.g., boxplots, histograms, heatmaps) to uncover patterns and correlations between features and churn.
# 6.) Create a fully working Dashboard
# 7.) Provide Actionable Recommendations: Derive insights from the analysis to suggest targeted retention strategies for the bank.


import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import norm



data = pd.read_csv('Bank_Churn.csv')



data
data.head()


# Multicollinearity Check with Variance Inflation Factor (VIF)
# Encode categorical variables (optional)
data_encoded = pd.get_dummies(data, drop_first=True)
# Keep only numeric data
numeric_data = data_encoded.select_dtypes(include=[np.number])
# Drop rows with missing values
numeric_data = numeric_data.dropna()
# Add constant term
X = add_constant(numeric_data)
# Compute VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)




data.shape



data.info()




data.describe(include = 'all')





data.isnull().sum()




data.Geography.nunique()





data.Geography.unique()




# Counting Number of active members
activeMembers = data[data['IsActiveMember'] == 1].shape[0]
print(activeMembers)
# Since, number of active members are equal to 5151. Hence the ratio of activeMembers/non-activeMembers is 0.5151.



maleCount = data[data['Gender'] == 'Male'].shape[0]
print(maleCount)




active_members = data[data["IsActiveMember"] == 1]
gender_counts = active_members["Gender"].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=gender_counts.index, y=gender_counts.values)
for i, value in enumerate(gender_counts.values):  
    plt.text(i, value + 50, str(value), ha='center', fontweight='bold')
plt.title("Number of Active Members by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Active Members")
plt.tight_layout()
plt.show()
# Detailed explanation:
# i.) enumerate(gender_counts.values):
#     --> gender_counts.values contains the heights of each bar (the count values)
#     --> enumerate() adds a counter (index) to these values, giving us both:
#         --> i: the index of the current bar (0 for first bar, 1 for second, etc.)
#         --> value: the height/value of the current bar
# ii.) plt.text(i, value + 50, str(value), ha='center', fontweight='bold'):
#     --> Adds text at position (i, value + 50)
#         --> i: x-coordinate (bar index)
#         --> value + 50: y-coordinate (slightly above the top of the bar)
#     --> str(value): The text to display (the bar's value converted to string)
#     --> ha='center': Horizontal alignment is centered
#     --> fontweight='bold': Makes the text bold
# Since, total active members are 5151. In 5151, 2867 belongs to male gender category and the rest 2284 belongs to female.




plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='Exited', y='CreditScore', data=data)
plt.title('Credit Score Distribution by Churn Status')
plt.subplot(1, 2, 2)
sns.histplot(data=data, x='CreditScore', hue='Exited', kde=True, element='step')
plt.title('Credit Score Distribution')
plt.show()
# 1: The customer has exited (churned) or closed their account with the bank.
# 0: The customer has not exited and remains an active customer of the bank.

# Boxplot Insights (Left Plot)<br>
# i.) The median credit scores for both churned (Exited = 1) and non-churned (Exited = 0) customers are quite similar.<br>
# ii.) Both groups have a similar interquartile range (middle 50% of data), though the churned group has slightly more lower-end outliers (credit scores below ~400).<br>
# iii.) Overall distribution spread is comparable, suggesting that credit score alone may not strongly distinguish churn behavior.
# <br>
# 
# Histogram & KDE Insights (Right Plot)<br>
# i.) The non-churned customers (Exited = 0) show a higher count across most credit score ranges, especially around the 650–700 mark.<br>
# ii.) The churned group (Exited = 1) has a more flattened distribution, indicating fewer customers with high credit scores.<br>
# iii.) There is a slight tendency for churned customers to have slightly lower credit scores on average, but the overlap is significant.



plt.figure(figsize=(10, 6))
sns.jointplot(x='Age', y='CreditScore', data=data, hue='Exited', kind='kde', palette={0: 'blue', 1: 'red'})
plt.suptitle('CreditScore vs. Age by Churn Status', y=1.02)
plt.show()
# plt.figure(figsize=(10, 6)) - Creates a figure with width 10 inches and height 6 inches
# sns.jointplot() - Creates a joint plot (combination plot) with:
# x='Age' - Age on the x-axis
# y='CreditScore' - CreditScore on the y-axis
# data=data - Uses the DataFrame named 'data'
# hue='Exited' - Colors the plot elements based on the 'Exited' column (churn status)
# kind='kde' - Uses Kernel Density Estimation for the visualization (shows density contours)
# palette={0: 'blue', 1: 'red'} - Uses blue for retained customers (0) and red for churned customers (1)
# plt.suptitle() - Adds a title at the top of the plot ("CreditScore vs. Age by Churn Status")
# y=1.02 - Adjusts the title position slightly above the plot
# plt.show() - Displays the plot

# Conclusion:<br>
# i.) CreditScore Distribution:<br>
# a.) Most customers (both retained and churned) have credit scores between 600-800<br>
# b.) The density appears slightly higher in the 650-750 range<br>
# ii.) Age Distribution:<br>
# a.) Retained customers (blue) show a relatively even age distribution<br>
# b.) Churned customers (red) appear more concentrated in older age groups (likely 40+)<br>
# iii.) Relationship Patterns:<br>
# a.) Older customers with mid-range credit scores (600-700) show higher churn density<br>
# b.) Younger customers across all credit scores are more likely to be retained<br>
# c.) Very high credit scores (>800) show lower churn regardless of age<br>
# 
# Recommendation:<br>
# Targeted Marketing: Middle-aged customers (30–50) with moderate credit scores (600–750) form the largest segment—ideal for retention strategies.




avg_credit_by_geo = data.groupby('Geography')['CreditScore'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Geography', y='CreditScore', data=avg_credit_by_geo, palette='viridis')
plt.title('Average Credit Score by Geography', fontsize=16)
plt.xlabel('Geography', fontsize=12)
plt.ylabel('Average Credit Score', fontsize=12)
plt.ylim(600, 700)
for index, row in avg_credit_by_geo.iterrows():
    plt.text(index, row['CreditScore'], f"{row['CreditScore']:.1f}", 
             ha='center', va='bottom', fontsize=12)
plt.show()
# Conclusion:<br>
# i.) Germany has the highest average credit score (≈ 650), suggesting customers in Germany tend to have better creditworthiness compared to other regions.<br>
# ii.) France follows closely (≈ 640), indicating relatively good credit profiles among French customers.<br>
# iii.) Spain has the lowest average credit score (≈ 630), which may imply higher credit risk or different lending standards in this region.




plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='Exited', y='Tenure', data=data)
plt.title('Tenure Distribution by Status')
plt.subplot(1, 2, 2)
sns.countplot(x='Tenure', hue='Exited', data=data)
plt.title('Tenure Distribution')
plt.show()
# Conclusion:<br>
# i.) Tenure is not a strong predictor of churn on its own.<br>
# ii.) Customers churn at all stages of their relationship with the bank.<br>
# iii.) Might be more valuable when combined with other features (e.g., products, activity status).




plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='Exited', y='EstimatedSalary', data=data)
plt.title('Salary Distribution by Status')
plt.subplot(1, 2, 2)
sns.histplot(data=data, x='EstimatedSalary', hue='Exited', kde=True, element='step')
plt.title('Salary Distribution')
plt.show()
# ✅ Conclusion:<br>
# i.) Estimated Salary is not a strong predictor of churn.<br>
# ii.) Churn is independent of salary level — customers across low, medium, and high salary brackets behave similarly.<br>
# iii.) This feature is likely not important for churn prediction on its own.




projects_by_gender = data.groupby('Gender')['NumOfProducts'].sum().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='Gender', y='NumOfProducts', data=projects_by_gender, palette='Set2')
plt.title('Total Number of Products by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Total Number of Products', fontsize=12)
plt.tight_layout()
plt.show()




plt.figure(figsize=(10, 6))
sns.barplot(x='Geography', y='Exited', hue='Gender', data=data)
plt.title('Churn Rate by Geography and Gender')
plt.show()





plt.figure(figsize=(10, 8))
corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Numerical Features')
plt.show()
# Conclusion:<br>
# i.) Churn is more behavioral (activity, product use) than financial (salary, credit score).<br>
# ii.) Age is the most influential positive correlate of churn.<br>
# Being active and engaged with multiple products can reduce churn.<br>

# Recommendations for the Bank:<br>
# i.) Targeted Retention Programs: Focus on older customers and customers of age between 30 to 40, especially females in Germany<br>
# 
# ii.) Product Bundling: Encourage customers to have 2-3 products (where churn is lowest)<br>
# 
# iii.) Engagement Strategies: Increase engagement with inactive members to reduce churn<br>
# 
# iv.) High-Balance Customers: Develop special programs for customers with high balances who may be more sensitive to service quality<br>



# # DASHBOARD


# --- Sample Dataset (for testing if you don't have 'data') ---
# Comment this out if you have your own dataset
np.random.seed(42)
data = pd.DataFrame({
    'Geography': np.random.choice(['France', 'Germany', 'Spain'], 1000),
    'Gender': np.random.choice(['Male', 'Female'], 1000),
    'Age': np.random.randint(18, 90, 1000),
    'Balance': np.random.uniform(0, 250000, 1000),
    'CreditScore': np.random.randint(300, 850, 1000),
    'EstimatedSalary': np.random.uniform(10000, 200000, 1000),
    'IsActiveMember': np.random.choice([0, 1], 1000),
    'HasCrCard': np.random.choice([0, 1], 1000),
    'Tenure': np.random.randint(0, 10, 1000),
    'NumOfProducts': np.random.randint(1, 4, 1000),
    'Exited': np.random.choice([0, 1], 1000)
})

# --- Data Preprocessing ---
# Make the data more readable and group numeric columns for analysis
data['Exited'] = data['Exited'].map({0: 'Retained', 1: 'Churned'})  # Convert 0/1 to Retained/Churned
data['IsActiveMember'] = data['IsActiveMember'].map({0: 'No', 1: 'Yes'})  # Convert 0/1 to No/Yes
data['HasCrCard'] = data['HasCrCard'].map({0: 'No', 1: 'Yes'})  # Convert 0/1 to No/Yes

# Group ages into ranges (e.g., 18-29, 30-39) for easier analysis
data['AgeGroup'] = pd.cut(data['Age'], 
                          bins=[18, 30, 40, 50, 60, 70, 80, 90],
                          labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
                          right=False)

# Group balances into ranges (e.g., 0-50k, 50-100k) for easier analysis
data['BalanceGroup'] = pd.cut(data['Balance'], 
                              bins=[-1, 0, 50000, 100000, 150000, 200000, float('inf')],
                              labels=['No Balance', '0-50k', '50-100k', '100-150k', '150-200k', '200k+'])

# --- Utility Functions ---
def filter_data(data, geo, gender, active, card, age):
    """Filter the dataset based on dropdown selections."""
    filtered = data.copy()
    if geo != 'All':
        filtered = filtered[filtered['Geography'] == geo]
    if gender != 'All':
        filtered = filtered[filtered['Gender'] == gender]
    if active != 'All':
        filtered = filtered[filtered['IsActiveMember'] == active]
    if card != 'All':
        filtered = filtered[filtered['HasCrCard'] == card]
    if age != 'All':
        filtered = filtered[filtered['AgeGroup'] == age]
    return filtered

def no_data_figure(title):
    """Create a placeholder chart when no data is available."""
    fig = go.Figure()
    fig.add_annotation(text="No data available for the selected filters",
                       xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                       font=dict(size=16))
    fig.update_layout(title_text=title, title_x=0.5, title_font_size=18)
    return fig

# --- Widgets (Interactive Controls) ---
# Dropdowns for filtering the data
geo_dropdown = widgets.Dropdown(options=['All'] + sorted(list(data['Geography'].unique())), 
                               description='Geography:', style={'description_width': 'initial'})
gender_dropdown = widgets.Dropdown(options=['All'] + sorted(list(data['Gender'].unique())), 
                                  description='Gender:', style={'description_width': 'initial'})
active_dropdown = widgets.Dropdown(options=['All', 'Yes', 'No'], 
                                  description='Active:', style={'description_width': 'initial'})
card_dropdown = widgets.Dropdown(options=['All', 'Yes', 'No'], 
                                description='Credit Card:', style={'description_width': 'initial'})
age_dropdown = widgets.Dropdown(options=['All'] + list(data['AgeGroup'].cat.categories), 
                               description='Age Group:', style={'description_width': 'initial'})
reset_button = widgets.Button(description='🔁 Reset Filters', button_style='warning', 
                             tooltip='Reset all filters to show all data')

# Output area for displaying charts and metrics
output = widgets.Output()

# Reset filters to default ("All")
def reset_filters(b):
    """Reset all dropdowns to 'All' when the reset button is clicked."""
    geo_dropdown.value = 'All'
    gender_dropdown.value = 'All'
    active_dropdown.value = 'All'
    card_dropdown.value = 'All'
    age_dropdown.value = 'All'

reset_button.on_click(reset_filters)

# --- Dashboard Update Function ---
def update_dashboard(change):
    """Update the dashboard when filters change."""
    with output:
        clear_output(wait=True)
        
        # Apply filters to the data
        filtered_data = filter_data(data, geo_dropdown.value, gender_dropdown.value, 
                                    active_dropdown.value, card_dropdown.value, age_dropdown.value)

        # Display dashboard title and key metrics
        display(Markdown("## 📊 Customer Churn Dashboard"))
        display(Markdown("### 📌 Key Metrics"))

        # Show total customers, churn rate, avg credit score, and avg age
        print(f"Total Customers: {len(filtered_data):,}")
        churn_rate = filtered_data['Exited'].value_counts(normalize=True).get('Churned', 0)
        print(f"Churn Rate: {churn_rate:.1%}")
        avg_credit = filtered_data['CreditScore'].mean()
        print(f"Average Credit Score: {avg_credit:.0f}" if not pd.isna(avg_credit) else "Average Credit Score: N/A")
        avg_age = filtered_data['Age'].mean()
        print(f"Average Age: {avg_age:.1f} years" if not pd.isna(avg_age) else "Average Age: N/A")

        # --- Churn by Category ---
        display(Markdown("### 📈 Churn by Category"))
        if filtered_data.empty:
            fig = no_data_figure("Churn Rate by Category")
        else:
            # Create a 2x3 grid of bar charts
            fig = make_subplots(rows=2, cols=3, 
                                subplot_titles=("Geography", "Gender", "Active Member", 
                                                "Credit Card", "Age Group", "Balance Group"),
                                vertical_spacing=0.15)
            # Add bar charts for each category
            for i, (col, row, col_num) in enumerate([
                ('Geography', 1, 1), ('Gender', 1, 2), ('IsActiveMember', 1, 3),
                ('HasCrCard', 2, 1), ('AgeGroup', 2, 2), ('BalanceGroup', 2, 3)
            ]):
                df = filtered_data.groupby([col, 'Exited']).size().unstack().fillna(0)
                df = df.div(df.sum(axis=1), axis=0) * 100  # Convert to percentage
                fig.add_trace(go.Bar(x=df.index.astype(str), y=df['Churned'], name=col), 
                              row=row, col=col_num)
            fig.update_layout(height=800, title_text="Churn Rate by Category", title_x=0.5, showlegend=False)
            fig.update_yaxes(title_text="Churn Rate (%)")
            fig.update_xaxes(tickangle=45)
        fig.show()

        # --- Numeric Features by Churn ---
        display(Markdown("### 📉 Numeric Features by Churn"))
        if filtered_data.empty:
            fig = no_data_figure("Numeric Features by Churn")
        else:
            # Create a 2x2 grid of box plots
            fig = make_subplots(rows=2, cols=2, 
                                subplot_titles=("Credit Score", "Age", "Balance", "Estimated Salary"))
            for col, row, col_num in [
                ('CreditScore', 1, 1), ('Age', 1, 2), ('Balance', 2, 1), ('EstimatedSalary', 2, 2)
            ]:
                fig.add_trace(go.Box(x=filtered_data['Exited'], y=filtered_data[col], name=col, boxmean=True), 
                              row=row, col=col_num)
            fig.update_layout(height=600, title_text="Numeric Features by Churn Status", title_x=0.5, showlegend=False)
        fig.show()

        # --- Balance vs Estimated Salary ---
        display(Markdown("### 💸 Balance vs Estimated Salary"))
        if filtered_data.empty:
            fig = no_data_figure("Balance vs Estimated Salary")
        else:
            # Create a scatter plot
            fig = px.scatter(filtered_data, x="EstimatedSalary", y="Balance", color="Exited",
                             title="Balance vs Estimated Salary",
                             hover_data=['Age', 'Geography', 'Gender'], opacity=0.6)
            fig.update_layout(height=600)
        fig.show()

        # --- Churn by Tenure ---
        display(Markdown("### 📅 Churn Rate by Tenure"))
        if filtered_data.empty:
            fig = no_data_figure("Churn Rate by Tenure")
        else:
            # Create a line chart
            tenure_df = filtered_data.groupby(['Tenure', 'Exited']).size().unstack().fillna(0)
            tenure_df = tenure_df.div(tenure_df.sum(axis=1), axis=0) * 100
            fig = px.line(tenure_df, x=tenure_df.index, y='Churned',
                          title="Churn Rate by Tenure", labels={'value': 'Churn Rate (%)'})
            fig.update_layout(height=400, showlegend=False)
        fig.show()

        # --- Churn by Number of Products ---
        display(Markdown("### 🛍️ Churn by Number of Products"))
        if filtered_data.empty:
            fig = no_data_figure("Churn Rate by Number of Products")
        else:
            # Create a bar chart
            products_df = filtered_data.groupby(['NumOfProducts', 'Exited']).size().unstack().fillna(0)
            products_df = products_df.div(products_df.sum(axis=1), axis=0) * 100
            fig = px.bar(products_df, x=products_df.index, y='Churned',
                         title="Churn by Number of Products",
                         labels={'value': 'Churn Rate (%)', 'NumOfProducts': 'Number of Products'})
            fig.update_layout(height=400, showlegend=False)
        fig.show()

# --- Connect Widgets to Update Function ---
# Update the dashboard when any dropdown changes
geo_dropdown.observe(update_dashboard, names='value')
gender_dropdown.observe(update_dashboard, names='value')
active_dropdown.observe(update_dashboard, names='value')
card_dropdown.observe(update_dashboard, names='value')
age_dropdown.observe(update_dashboard, names='value')

# --- Display the Dashboard ---
# Arrange filters in a clean layout
filters_box = widgets.VBox([
    widgets.HTML(value="<h3>🔎 Filters</h3>"),
    widgets.HBox([geo_dropdown, gender_dropdown]),
    widgets.HBox([active_dropdown, card_dropdown, age_dropdown]),
    reset_button
])

# Show the filters and output area
display(filters_box, output)
update_dashboard(None)






