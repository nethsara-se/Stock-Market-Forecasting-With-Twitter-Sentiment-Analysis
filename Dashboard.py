import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError

# Ensure the default encoding is UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')

# Function to load models
def load_models(company):
    mse = MeanSquaredError()
    MLP_Tweet = load_model(f'MLP_{company}_Tweet.h5', custom_objects={'mse': mse}, compile=False)
    MLP_Stock = load_model(f'MLP_{company}_Stock.h5', custom_objects={'mse': mse}, compile=False)
    return MLP_Tweet, MLP_Stock

# Function to scale values
def scalevalue(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data

# Function to split sequences
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Load the dataset
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file, encoding='utf-8')

# Streamlit app
st.markdown('<h1 style="font-size: 36px;">Stock Market Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Dropdown menu for company selection
company = st.sidebar.selectbox("Select Company", ["Apple", "Google", "Microsoft", "Amazon", "Tesla"])

st.markdown('<h2 style="font-size: 16px;">Upload Dataset</h2>', unsafe_allow_html=True)
# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
if uploaded_file:
    data = load_data(uploaded_file)
    st.write("", data.head())

    # Sentiment dataset
    dataset = data[['day_date', 'polarity_score', 'negative', 'neutral', 'positive', 'open_value', 'high_value', 'low_value', 'volume', 'stock_price']]
    company_df_test = dataset
    dataset = dataset.drop(columns=['day_date'])

    # Scale the dataset
    scaled_dataset = scalevalue(dataset)

    # Time steps
    n_steps_in, n_steps_out = 60, 20

    # Generate input/output for testing sets
    X_test_company, y_test_company = split_sequences(scaled_dataset, n_steps_in, n_steps_out)

    # Flatten input
    n_input = X_test_company.shape[1] * X_test_company.shape[2]
    X_test_company_flatteded = X_test_company.reshape((X_test_company.shape[0], n_input))

    # Non-sentiment dataset
    dataset_stock = data[['day_date', 'open_value', 'high_value', 'low_value', 'volume', 'stock_price']]
    dataset_stock = dataset_stock.drop(columns=['day_date'])

    # Scale the dataset
    scaled_dataset_stock = scalevalue(dataset_stock)

    # Generate input/output for testing sets
    X_test_company_stock, y_test_company_stock = split_sequences(scaled_dataset_stock, n_steps_in, n_steps_out)

    # Flatten input
    n_input_stock = X_test_company_stock.shape[1] * X_test_company_stock.shape[2]
    X_test_company_stock_flatteded = X_test_company_stock.reshape((X_test_company_stock.shape[0], n_input_stock))

    # Load models
    MLP_Tweet, MLP_Stock = load_models(company)

    # Predict stock prices using the models
    predicted_stock_prices_company = MLP_Tweet.predict(X_test_company_flatteded)
    predicted_stock_prices_company_stock = MLP_Stock.predict(X_test_company_stock_flatteded)

    # Extract the actual stock prices and the dates
    actual_stock_prices = company_df_test['stock_price'].values
    dates = pd.to_datetime(company_df_test['day_date']).values

    # Fit the scaler on the stock prices
    stock_price_scaler = MinMaxScaler(feature_range=(0, 1))
    stock_price_scaler.fit(dataset[['stock_price']])

    # Inverse transform the scaled predicted stock prices
    predicted_stock_prices_company_original = stock_price_scaler.inverse_transform(predicted_stock_prices_company)
    last_predicted_values_company = predicted_stock_prices_company_original[:, -1]

    # Ensure lengths match for plotting
    if len(last_predicted_values_company) < len(actual_stock_prices):
        actual_stock_prices = actual_stock_prices[:len(last_predicted_values_company)]
        dates = dates[:len(last_predicted_values_company)]
    elif len(last_predicted_values_company) > len(actual_stock_prices):
        last_predicted_values_company = last_predicted_values_company[:len(actual_stock_prices)]

    # Inverse transform for predicted_stock_prices_company_stock
    predicted_stock_prices_company_stock_original = stock_price_scaler.inverse_transform(predicted_stock_prices_company_stock)
    last_predicted_values_company_stock = predicted_stock_prices_company_stock_original[:, -1]

    # Ensure lengths match for plotting
    if len(last_predicted_values_company_stock) < len(actual_stock_prices):
        last_predicted_values_company_stock = last_predicted_values_company_stock[:len(actual_stock_prices)]
    elif len(last_predicted_values_company_stock) > len(actual_stock_prices):
        actual_stock_prices = actual_stock_prices[:len(last_predicted_values_company_stock)]
        dates = dates[:len(last_predicted_values_company_stock)]

    # Create subplots with a dark background
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Add a main title for the graphical view
    st.markdown('<h2 style="font-size: 16px;">Graphical View of Stock Prices</h2>', unsafe_allow_html=True)

    # Plot for predicted_stock_prices_company
    ax1.plot(dates, actual_stock_prices, color='mediumblue', label='Actual Stock Prices', linewidth=2)
    ax1.plot(dates, last_predicted_values_company, color='lightblue', label='Predicted Stock Prices (With Twitter)', linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price USD ($)')
    ax1.set_title(f'{company} Stock Prices - Actual vs Predicted (With Twitter)', fontsize=14)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(color='gray', linestyle='--', linewidth=0.5)

    # Plot for predicted_stock_prices_company_stock
    ax2.plot(dates, actual_stock_prices, color='mediumblue', label='Actual Stock Prices', linewidth=2)
    ax2.plot(dates, last_predicted_values_company_stock, color='violet', label='Predicted Stock Prices (Without Twitter)', linewidth=2)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Stock Price USD ($)')
    ax2.set_title(f'{company} Stock Prices - Actual vs Predicted (Without Twitter)', fontsize=14)
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(color='gray', linestyle='--', linewidth=0.5)

    # Adjust x-axis ticks
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Show the plot
    st.pyplot(fig)
