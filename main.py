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

st.title("Stock & Currency Prediction App")

# Updated stock list - removed META, AMZN, NFLX, MSFT and added KRW-USD
stocks = ("AAPL", "GOOG", "GME", "NVDA", "TSLA", "BTC-USD", "DOGE-USD", "KRW=X")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# Updated START_DATES dictionary
START_DATES = {
    'AAPL': '2010-01-01',
    'GOOG': '2010-01-01',
    'GME': '2010-01-01',
    'NVDA': '2010-01-01',
    'TSLA': '2010-01-01',
    'BTC-USD': '2014-01-01',
    'DOGE-USD': '2021-01-01',
    'KRW=X': '2010-01-01'
}

# Slider for the prediction period
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Adjust START date based on the selected stock to ensure data availability
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['Open'], 
        name='Open',
        line=dict(color='#1f77b4', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['Close'], 
        name='Close',
        line=dict(color='#d62728', width=1)
    ))
    fig.layout.update(
        title_text="Time Series Data", 
        xaxis_rangeslider_visible=True,
        height=600,
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)

# Add progress indicators
with st.spinner('Loading data...'):
    data = load_data(selected_stock, START, TODAY)
    
    if data.empty:
        st.error("No data downloaded. Please check the ticker symbol or date range.")
        st.stop()

# Show raw data in an expander
with st.expander("Show Raw Data"):
    st.write(data.tail())

# Plot the time series data
st.subheader("Historical Price Chart")
plot_raw_data()

# Add a separator
st.markdown("---")

# Show the forecast section in a container
st.subheader("Forecast Section")
with st.container():
    with st.spinner('Generating forecast...'):
        # Prepare data for Prophet
        df_train = pd.DataFrame()
        df_train['ds'] = data['Date']
        df_train['y'] = data['Close']

        # Clean data
        df_train['ds'] = pd.to_datetime(df_train['ds'])
        df_train['y'] = pd.to_numeric(df_train['y'])
        df_train = df_train.dropna()

        # Initialize and fit Prophet model
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )

        # Fit the model
        try:
            m.fit(df_train)
        except Exception as e:
            st.error(f"Error fitting the model: {e}")
            st.stop()

        # Create future dates and forecast
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Plot forecast data
        st.subheader("Forecast Plot")
        fig1 = plot_plotly(m, forecast)
        fig1.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Display forecast components
        st.subheader("Forecast Components")
        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)
# Display forecast components
st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)

