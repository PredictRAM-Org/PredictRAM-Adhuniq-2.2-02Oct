import os
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

# Load Data
@st.cache_data
def load_data():
    # Load financial and synthetic data
    iip_data = pd.read_excel('IIP2024.xlsx')
    synthetic_data = pd.read_excel('Synthetic_Industry_Data.xlsx', sheet_name=None)

    # Load stock data
    stock_data = {}
    stock_data_folder = 'stockdata'
    for filename in os.listdir(stock_data_folder):
        if filename.endswith('.csv'):
            stock_name = filename.replace('.csv', '')
            stock_data[stock_name] = pd.read_csv(os.path.join(stock_data_folder, filename))

    # Load correlation results, including new columns
    correlation_results = pd.read_excel(os.path.join(stock_data_folder, 'Manufacture_of_Food_Products_correlation_results.xlsx'))

    # Load financial data
    financial_data = {}
    financial_folder = 'financial'
    for filename in os.listdir(financial_folder):
        if filename.endswith('.xlsx'):
            stock_name = filename.replace('.xlsx', '')
            stock_file_path = os.path.join(financial_folder, filename)
            financial_data[stock_name] = pd.read_excel(stock_file_path, sheet_name=None)

    return iip_data, synthetic_data, stock_data, correlation_results, financial_data

# Define Industry and Indicators
def define_industry_indicators():
    indicators = {
        'Manufacture of Food Products': {
            'Leading': ['Consumer Spending Trends', 'Agricultural Output', 'Retail Sales Data'],
            'Lagging': ['Inventory Levels', 'Employment Data']
        },
        # Add other industries and their indicators here
    }
    return indicators

# Interpret correlation values based on magnitude
def interpret_correlation(value):
    if value > 0.8:
        return "Strong Positive"
    elif 0.3 < value <= 0.8:
        return "Slight Positive"
    elif -0.3 <= value <= 0.3:
        return "Neutral"
    elif -0.8 <= value < -0.3:
        return "Slight Negative"
    else:
        return "Strong Negative"

# Linear Regression for forecasting
def linear_regression_prediction(X, y):
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict(X)
    return prediction, model

# ARIMA for time series forecasting
def arima_prediction(y, order=(1, 1, 1)):
    model = ARIMA(y, order=order)
    model_fit = model.fit()
    prediction = model_fit.forecast(steps=5)
    return prediction, model_fit

# Random Forest for regression
def random_forest_prediction(X, y):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    prediction = model.predict(X)
    return prediction, model

# Calculate and display correlation results
def display_correlation_results(selected_stocks, correlation_results, financial_data):
    all_adjusted_corr_data = []
    
    for stock in selected_stocks:
        st.subheader(f'Correlation Analysis for {stock}')

        stock_correlation_data = correlation_results[correlation_results['Stock Name'] == stock]

        if not stock_correlation_data.empty:
            st.write('**Actual Correlation Results:**')
            st.write(stock_correlation_data)

            st.subheader('Predicted Correlation Analysis')

            updated_corr_data = stock_correlation_data.copy()

            # Adjust and interpret new columns
            for col in [
                'Total Revenue/Income Coefficient', 'Total Operating Expense Coefficient', 
                'Operating Income/Profit Coefficient', 'EBITDA Coefficient', 'EBIT Coefficient',
                'Income/Profit Before Tax Coefficient', 'Net Income From Continuing Operation Coefficient',
                'Net Income Coefficient', 'Net Income Applicable to Common Share Coefficient',
                'EPS (Earning Per Share) Coefficient', 'Operating Margin Coefficient', 'EBITDA Margin Coefficient',
                'Net Profit Margin Coefficient'
            ]:
                if col in updated_corr_data.columns:
                    # Example logic: adjust coefficient by 5%
                    updated_corr_data[f'Adjusted {col}'] = updated_corr_data[col] * 1.05
                    updated_corr_data[f'Interpreted {col}'] = updated_corr_data[col].apply(lambda x: "Positive" if x > 0 else "Negative")

            # Add Standard Errors, t-Statistics, P-Values, and Confidence Intervals
            for metric in ['Total Revenue/Income', 'Total Operating Expense', 'Operating Income/Profit', 'EBITDA', 'EBIT', 'Income/Profit Before Tax', 'Net Income From Continuing Operation', 'Net Income', 'Net Income Applicable to Common Share', 'EPS (Earning Per Share)', 'Operating Margin', 'EBITDA Margin', 'Net Profit Margin']:
                standard_error_col = f'{metric} Standard Error'
                t_stat_col = f'{metric} t-Statistic'
                p_value_col = f'{metric} P-Value'
                confidence_interval_col = f'{metric} Confidence Interval'

                if all(col in updated_corr_data.columns for col in [standard_error_col, t_stat_col, p_value_col, confidence_interval_col]):
                    updated_corr_data[f'{metric} Interpretation'] = updated_corr_data.apply(
                        lambda row: f"Standard Error: {row[standard_error_col]}, t-Statistic: {row[t_stat_col]}, P-Value: {row[p_value_col]}, Confidence Interval: {row[confidence_interval_col]}", axis=1
                    )

            all_adjusted_corr_data.append(updated_corr_data)

    # Combine and display all predicted correlation data
    if all_adjusted_corr_data:
        combined_corr_data = pd.concat(all_adjusted_corr_data, ignore_index=True)
        st.write('**Predicted Correlation Results:**')
        st.write(combined_corr_data)

# Function to get detailed interpretation with Indian economy context
def get_detailed_interpretation(parameter_name, correlation_interpretation):
    # Example custom interpretation
    interpretations = {
        "correlation with Total Revenue/Income": {
            "Slight Positive": (
                "* **Economic Context**: A slight increase in revenue indicates moderate market growth in India."
            ),
            "Strong Positive": (
                "* **Economic Context**: A strong increase in revenue highlights robust economic conditions."
            ),
        },
        # Add more parameter interpretations...
    }
    return interpretations.get(parameter_name, {}).get(correlation_interpretation, "No interpretation available.")

# Streamlit App Interface
st.title('Industry and Financial Data Prediction')

# Sidebar for industry selection
selected_industry = st.sidebar.selectbox('Select Industry', ['Manufacture of Food Products', 'Manufacture of Beverages'])

# Load all necessary data
iip_data, synthetic_data, stock_data, correlation_results, financial_data = load_data()

# Sidebar for stock selection
selected_stocks = st.sidebar.multiselect('Select Stocks', correlation_results['Stock Name'].tolist())

if selected_stocks:
    display_correlation_results(selected_stocks, correlation_results, financial_data)

    # Example of applying models to the stock data
    for stock in selected_stocks:
        st.subheader(f'Model Predictions for {stock}')
        # Assuming financial_data contains X and y for each stock

        # Linear Regression Example
        st.write('**Linear Regression Prediction**')
        X = financial_data[stock]['X_column']  # Replace 'X_column' with the actual feature
        y = financial_data[stock]['y_column']  # Replace 'y_column' with the target variable
        lr_prediction, lr_model = linear_regression_prediction(X, y)
        st.write(lr_prediction)

        # ARIMA Example
        st.write('**ARIMA Prediction**')
        arima_pred, arima_model = arima_prediction(y)
        st.write(arima_pred)

        # Random Forest Example
        st.write('**Random Forest Prediction**')
        rf_prediction, rf_model = random_forest_prediction(X, y)
        st.write(rf_prediction)
