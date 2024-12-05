import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# Set default start date for historical data
DEFAULT_START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Webapp by Jakhongir")

# Dropdown menu for stock selection
stocks = ("AAPL", "GOOG", "MSFT", "GME", "NVDA", "TSLA", "NFLX", "BTC-USD", "META", "AMZN", "DOGE-USD")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# Slider for the prediction period
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Adjust START date based on the selected stock to ensure data availability
START_DATES = {
    'AAPL': '2010-01-01',
    'GOOG': '2010-01-01',
    'MSFT': '2010-01-01',
    'GME': '2010-01-01',
    'NVDA': '2010-01-01',
    'TSLA': '2010-01-01',
    'NFLX': '2010-01-01',
    'BTC-USD': '2014-01-01',
    'META': '2012-05-18',  # Facebook IPO date
    'AMZN': '2010-01-01',
    'DOGE-USD': '2021-01-01',  # Data for DOGE-USD may not be available before 2021
}

START = START_DATES.get(selected_stock, DEFAULT_START)

# Function to load data
@st.cache_data  # Use @st.cache if your Streamlit version is below 1.18.0
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    return data

# Load and display data
data_load_state = st.text("Loading data...")
data = load_data(selected_stock, START, TODAY)

# Check if data is empty
if data.empty:
    st.error("No data downloaded. Please check the ticker symbol or date range.")
    st.stop()

data_load_state.text("Loading data... done!")

st.subheader("Raw data")
st.write(data.tail())

# Function to plot raw data
def plot_raw_data():
    # Determine the correct date column
    if 'Date' in data.columns:
        date_col = 'Date'
    elif 'Datetime' in data.columns:
        date_col = 'Datetime'
    else:
        st.error("Date column not found in data")
        st.stop()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[date_col], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data[date_col], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for Prophet
# Ensure 'Date' column exists in data
if 'Date' in data.columns:
    date_col = 'Date'
elif 'Datetime' in data.columns:
    date_col = 'Datetime'
else:
    st.error("Date column not found in data")
    st.stop()

# Copy the date column
df_train = data[[date_col]].copy()

# Handle multi-level columns for 'Close'
if 'Close' in data.columns:
    df_train['y'] = data['Close']
elif isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns.get_level_values(0):
    # Extract the first column under 'Close' in a multi-index
    df_train['y'] = data['Close'].iloc[:, 0].values
else:
    st.error("Close column not found in data")
    st.stop()

# Rename date_col to 'ds' for Prophet
df_train.rename(columns={date_col: 'ds'}, inplace=True)

# Debugging the initial DataFrame
st.write("Debugging df_train before cleaning:")
st.write(df_train.head())
st.write("Data types in df_train:")
st.write(df_train.dtypes)

# Ensure 'ds' and 'y' are in df_train
st.write("Columns in df_train before cleaning:")
st.write(df_train.columns.tolist())

# Clean data
df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')  # Ensure datetime format
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')     # Ensure numeric values

# Check for NaN values in 'ds' and 'y'
st.write("Number of NaN values in 'ds':", df_train['ds'].isna().sum())
st.write("Number of NaN values in 'y':", df_train['y'].isna().sum())

# Drop rows with NaN in 'ds' or 'y'
df_train.dropna(subset=['ds', 'y'], inplace=True)

# Final debug
st.write("Cleaned df_train:")
st.write(df_train.head())
st.write("Data types in df_train after cleaning:")
st.write(df_train.dtypes)

# Ensure there is sufficient data after cleaning
if df_train.empty:
    st.error("No data available for the selected stock and date range after cleaning. Please select a different stock or adjust the date range.")
    st.stop()

# Initialize and fit Prophet model
m = Prophet()

# Fit the model
try:
    m.fit(df_train)
except Exception as e:
    st.error(f"Error fitting the model: {e}")
    st.stop()

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

