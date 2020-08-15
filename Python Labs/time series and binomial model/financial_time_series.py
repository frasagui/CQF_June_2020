# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:39:26 2020

@author: xiaon
"""

# Import required libraries
import pandas as pd
import numpy as np

# Import data libraries
import yfinance as yf

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Fetch data by specifying the number of the period
spy=yf.download('SPY', period='5d')
print(spy.head())

# Fetch the data from yahoo using start and end dates
spy1=yf.download('SPY', start='2020-06-20' , end='2020-06-30', progress=False)
print(spy1)

# Define the parameters
faang_stocks = ['AAPL', 'AMZN', 'FB', 'GOOG', 'NFLX']
start_date = '2019-06-01'
end_date = '2020-06-30'

# Fetching data for multiple tickers
df = yf.download(faang_stocks, start=start_date, end=end_date, progress=False)['Adj Close']

# Display the first five rows of the dataframe to check the results. 
print(df.head())

# Let's save the data for future use
df.to_csv('faang_stocks_1.csv')

# # Display the first five rows of the dataframe to check the results. 
print(df.tail())

# Use comprehension to fetch data for multiple fields 
ohlc_data = {symbol: yf.download(symbol, start='2020-06-01', end='2020-06-30', progress=False) for symbol in faang_stocks}
print(ohlc_data)
print(ohlc_data.keys())

# Display AMZN stock data
print(ohlc_data['AMZN'].head())

# Display AMZN Adjusted Close data
print(ohlc_data['AMZN']['Adj Close'].head())

df.dtypes
df.index
df.info() # dataframe metadata information

# Retrive intraday data
dfi = yf.download(tickers="SPY", period="5d", interval="1m", progress=False)

# Display the first five rows of the dataframe to check the results.
print(dfi.head(20))

print(dfi.tail())

from alpha_vantage.timeseries import TimeSeries

key_path = "Key.txt"
ts = TimeSeries(key=open(key_path, 'r').read(),output_format='pandas')
data, metadata  = ts.get_intraday(symbol='AMZN',interval='1min', outputsize='full')

print(metadata)

print(data)

# Dataframe to Excel
from pandas import ExcelWriter
# Storing the fetched data in a separate sheet for each security
writer = ExcelWriter('mystocks_1.xlsx')

[pd.DataFrame(ohlc_data[symbol]).to_excel(writer,symbol) for symbol in faang_stocks]

writer.save() # save file

faang = pd.read_excel('mystocks_1.xlsx', sheet_name='NFLX',index_col=0, parse_dates=True)

# Display the last five rows of the data frame to check the results
faang.tail()

# Save ohlcv data for each securities in stockname.csv format
[pd.DataFrame(ohlc_data[symbol]).to_csv(symbol+'.csv') for symbol in faang_stocks]
print('*** data saved ***')

# Using get_history function to retriev Cboe Volatility Index time series for a specific dates
appl = pd.read_csv('AAPL.csv', index_col=0, parse_dates=True, dayfirst=True) 

# Display the last five rows of the data frame to check the results
print(appl.tail())

# Resampling to derive weekly values fro daily time series
spy_weekly = spy.resample('W').last()

# Resampling to derive monthly values from daily time series
spy_monthly = spy.resample('M').last()

# Display the first five rows of the data frame to check the output
print(spy_weekly.tail(5))

print(spy.index)

spy_weekly_thu = spy.resample('W-THU').ffill()
print(spy_weekly_thu.tail())

# Import cufflinks
import cufflinks as cf
cf.set_config_file(offline=True)

df['AAPL'].iplot(kind='line', title='CBOE Volatily Index')

import matplotlib.pyplot as plt
_=df['AAPL'].plot(kind='line', title='CBOE Volatily Index')
plt.show()



