import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly 
from plotly import graph_objs as go 


print("All imports are working!")

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Webapp by Jakhongir")

stocks = ("KRW", "AAPL", "GOOG", "MSFT", "GME", "NVDA", "TSLA", "NFLX", "BTC-USD", "META", "AMZN", "DOGE-USD")
selected_stocks  = st.selectbox("Select dataset fro prediction", stocks)

n_years = st.slider("Years of prediction:", 1 , 4)
period = n_years * 365
