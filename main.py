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
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load and display data
data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data... done!")

st.subheader("Raw data")
st.write(data.tail())

# Function to plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Clean data
df_train['ds'] = pd.to_datetime(df_train['ds'])  # Ensure datetime format
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')  # Ensure numeric values
df_train = df_train.dropna()  # Drop rows with invalid data

# Debugging output
st.subheader("Cleaned Data for Prophet")
st.write("Cleaned df_train:")
st.write(df_train.head())
st.write("Data types in df_train:")
st.write(df_train.dtypes)

# Initialize and fit Prophet model
m = Prophet()

# Additional debug info before model fitting
st.write("Inspecting df_train before fitting the model:")
st.write(df_train.head())
st.write("Data types of df_train:")
st.write(df_train.dtypes)

# Fit the model
m.fit(df_train)

# Create future dates and forecast
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader('Forecast Data')
st.write(forecast.tail())

# Plot forecast data
st.write("Forecast Plot")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Display forecast components
st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)
