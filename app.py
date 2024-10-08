import os
import pandas as pd
import streamlit as st

# Load Data
@st.cache_data
def load_data():
    # Load industry and financial data
    iip_data = pd.read_excel('IIP2024.xlsx')
    synthetic_data = pd.read_excel('Synthetic_Industry_Data.xlsx', sheet_name=None)

    # Load stock data
    stock_data = {}
    stock_data_folder = 'stockdata'
    for filename in os.listdir(stock_data_folder):
        if filename.endswith('.csv'):
            stock_name = filename.replace('.csv', '')
            stock_data[stock_name] = pd.read_csv(os.path.join(stock_data_folder, filename))

    # Load correlation results, including new added columns
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

# Define Industry and Indicators (you can customize based on industries)
def define_industry_indicators():
    indicators = {
        'Manufacture of Food Products': {
            'Leading': ['Consumer Spending Trends', 'Agricultural Output', 'Retail Sales Data'],
            'Lagging': ['Inventory Levels', 'Employment Data']
        },
        # You can continue adding more industries and indicators here...
    }
    return indicators

# Interpret correlation values based on their magnitude
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

# Function to calculate and display correlation results
def display_correlation_results(selected_stocks, correlation_results, financial_data):
    # Display and calculate actual and predicted correlation results
    all_adjusted_corr_data = []
    
    for stock in selected_stocks:
        st.subheader(f'Correlation Analysis with {stock}')

        stock_correlation_data = correlation_results[correlation_results['Stock Name'] == stock]

        if not stock_correlation_data.empty:
            st.write('**Actual Correlation Results:**')
            st.write(stock_correlation_data)

            st.subheader('Predicted Correlation Analysis')

            updated_corr_data = stock_correlation_data.copy()

            # Example logic to apply predictions and adjust columns
            for col in [
                'Total Revenue/Income Coefficient', 'Total Operating Expense Coefficient', 
                'Operating Income/Profit Coefficient', 'EBITDA Coefficient', 'EBIT Coefficient',
                'Income/Profit Before Tax Coefficient', 'Net Income From Continuing Operation Coefficient',
                'Net Income Coefficient', 'Net Income Applicable to Common Share Coefficient',
                'EPS (Earning Per Share) Coefficient', 'Operating Margin Coefficient', 'EBITDA Margin Coefficient',
                'Net Profit Margin Coefficient'
            ]:
                if col in updated_corr_data.columns:
                    updated_corr_data[f'Adjusted {col}'] = updated_corr_data[col] * 1.05  # Example logic for adjustment
                    updated_corr_data[f'Interpreted {col}'] = updated_corr_data[col].apply(lambda x: "Positive" if x > 0 else "Negative")

            all_adjusted_corr_data.append(updated_corr_data)

    # Combine and display all predicted correlation data
    if all_adjusted_corr_data:
        combined_corr_data = pd.concat(all_adjusted_corr_data, ignore_index=True)
        st.write('**Predicted Correlation Results:**')
        st.write(combined_corr_data)

# Function to get detailed interpretation with Indian economy context
def get_detailed_interpretation(parameter_name, correlation_interpretation):
    # You can customize this with context-specific data
    interpretations = {
        "correlation with Total Revenue/Income": {
            "Slight Positive": (
                "* **Economic Context**: A slight increase in revenue suggests moderate market growth in India."
            ),
            "Strong Positive": (
                "* **Economic Context**: A significant rise in revenue reflects robust economic conditions."
            ),
            # More interpretation based on economic parameters...
        },
        # Add more parameters and interpretations here...
    }
    return interpretations.get(parameter_name, {}).get(correlation_interpretation, "No interpretation available.")

# Streamlit App Interface
st.title('Industry and Financial Data Prediction')

# Sidebar for industry selection
selected_industry = st.sidebar.selectbox('Select Industry', ['Manufacture of Food Products', 'Manufacture of Beverages'])

# Load data
iip_data, synthetic_data, stock_data, correlation_results, financial_data = load_data()

# Define industry indicators (hidden part)
indicators = define_industry_indicators()

# Sidebar for stock selection
selected_stocks = st.sidebar.multiselect('Select Stocks', correlation_results['Stock Name'].tolist())

if selected_stocks:
    display_correlation_results(selected_stocks, correlation_results, financial_data)

