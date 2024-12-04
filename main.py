import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# Set start date for historical data
START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Webapp by Jakhongir")

# Dropdown menu for stock selection
stocks = ("AAPL", "GOOG", "MSFT", "GME", "NVDA", "TSLA", "NFLX", "BTC-USD", "META", "AMZN", "DOGE-USD")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

# Slider for the prediction period
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Function to load data
@st.cache_data  # Use @st.cache if your Streamlit version is below 1.18.0
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load and display data
data_load_state = st.text("Loading data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data... done!")

st.subheader("Raw data")
st.write(data.tail())

# Function to plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for Prophet
df_train = data[['Date']].copy()

# Handle multi-level columns for 'Close'
if isinstance(data.columns, pd.MultiIndex):
    # Extract the first column under 'Close'
    df_train['y'] = data['Close'].iloc[:, 0].values
else:
    df_train['y'] = data['Close']

# Rename 'Date' to 'ds' for Prophet
df_train.rename(columns={'Date': 'ds'}, inplace=True)

# Debugging the initial DataFrame
st.write("Debugging df_train before cleaning:")
st.write(df_train.head())
st.write("Data types in df_train:")
st.write(df_train.dtypes)

# Clean data
df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')  # Ensure datetime format
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')     # Ensure numeric values
df_train.dropna(subset=['ds', 'y'], inplace=True)                 # Drop rows with NaN in 'ds' or 'y'

# Final debug
st.write("Cleaned df_train:")
st.write(df_train.head())
st.write("Data types in df_train after cleaning:")
st.write(df_train.dtypes)

# Ensure there is sufficient data
if df_train.empty:
    st.error("No data available for the selected stock and date range. Please select a different stock or adjust the date range.")
else:
    # Initialize and fit Prophet model
    m = Prophet()

    # Fit the model
    m.fit(df_train)

    # Create future dates and forecast
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Display forecast data
    st.subheader("Forecast Data")
    st.write(forecast.tail())

    # Plot forecast data
    st.write("Forecast Plot")
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # Display forecast components
    st.write("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)
