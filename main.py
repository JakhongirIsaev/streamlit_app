import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly 
from plotly import graph_objs as go 


START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Webapp by Jakhongir")

stocks = ("KRW", "AAPL", "GOOG", "MSFT", "GME", "NVDA", "TSLA", "NFLX", "BTC-USD", "META", "AMZN", "DOGE-USD")
selected_stocks  = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1 , 4)
period = n_years * 365

@st.cashe
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data 

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data.. done!")

st.subheader("Raw data")
st.write(data.tail)
