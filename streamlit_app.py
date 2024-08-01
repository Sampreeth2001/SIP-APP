import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Time Series Analysis App", layout="wide")

# Streamlit input fields
st.title("Time Series Analysis App")
ticker = st.text_input("Enter the stock ticker:", "AAPL")
start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End date", value=pd.to_datetime("2023-01-01"))

# Download data
if st.button("Download Data"):
    data = yf.download(ticker, start=start_date, end=end_date)
    st.write("Data downloaded successfully!")
    st.dataframe(data)

    # Decompose the time series
    st.subheader("Time Series Decomposition")
    result = seasonal_decompose(data['Adj Close'], model='additive')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    result.observed.plot(ax=ax1, title='Observed')
    result.trend.plot(ax=ax2, title='Trend')
    result.seasonal.plot(ax=ax3, title='Seasonal')
    result.resid.plot(ax=ax4, title='Residual')
    st.pyplot(fig)

    # Split the data into train and test sets
    st.subheader("Train/Test Split")
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    st.write(f"Train data size: {len(train_data)}, Test data size: {len(test_data)}")

    # Fit a Random Forest model
    st.subheader("Random Forest Model")
    X_train = np.arange(len(train_data)).reshape(-1, 1)
    y_train = train_data['Adj Close']
    X_test = np.arange(len(train_data), len(data)).reshape(-1, 1)
    y_test = test_data['Adj Close']
    
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)
    
    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_data.index, y_train, label='Train Data')
    ax.plot(test_data.index, y_test, label='Test Data')
    ax.plot(test_data.index, predictions, label='Predictions')
    ax.legend()
    st.pyplot(fig)
    
    # Compute evaluation metrics
    st.subheader("Model Evaluation")
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    r2 = r2_score(y_test, predictions)
    
    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")
    st.write(f"MAPE: {mape}")
    st.write(f"R-squared: {r2}")

    # Conduct ADF test
    st.subheader("ADF Test")
    adf_result = adfuller(data['Adj Close'])
    st.write(f"ADF Statistic: {adf_result[0]}")
    st.write(f"p-value: {adf_result[1]}")
    for key, value in adf_result[4].items():
        st.write(f"Critical Value {key}: {value}")

    # Fit VECM model
    st.subheader("VECM Model")
    diff_data = data.diff().dropna()
    vecm_model = VECM(diff_data, k_ar_diff=1, coint_rank=1)
    vecm_fit = vecm_model.fit()
    st.write("VECM Model Summary:")
    st.text(vecm_fit.summary())

    # Fit VAR model
    st.subheader("VAR Model")
    var_model = VAR(diff_data)
    var_fit = var_model.fit(maxlags=1)
    st.write("VAR Model Summary:")
    st.text(var_fit.summary())

    # Plot VECM and VAR results
    st.subheader("VECM and VAR Model Plots")
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    vecm_fit.plot_forecast(steps=10, ax=ax[0])
    ax[0].set_title("VECM Forecast")
    var_fit.plot_forecast(steps=10, ax=ax[1])
    ax[1].set_title("VAR Forecast")
    st.pyplot(fig)
