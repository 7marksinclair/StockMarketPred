import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from datetime import datetime

import yfinance as yf


current_datetime = datetime.now()

st.title('Stock Prediction')
st.write('Today--->')
st.write(current_datetime.date())
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

tickerData = yf.Ticker(user_input)
df = tickerData.history(period='1d', start='2021-01-01', end=current_datetime.date())

#Describe_data

st.subheader('Data from 2021 to Present')
st.write(df.describe())

#visualize

st.subheader('Closing price vs time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs time chart w 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs time chart w 100MA, 200MA')
ma200 = df.Close.rolling(200).mean()
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200, 'g')
plt.plot(ma100, 'r')
plt.plot(df.Close)
st.pyplot(fig)

#Spliting data

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
st.subheader('Training chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data_training.Close)
st.pyplot(fig)

data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
st.subheader('Testing chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data_testing.Close)
st.pyplot(fig)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

#Loading model

model = load_model('keras_model.h5', custom_objects={'CustomAdam': CustomAdam})

#Testing part

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

latest_data = tickerData.history(period='1d')
current_price = latest_data['Close'].iloc[-1]
mul_factor = float(current_price/y_predicted[y_predicted.shape[0]-1][0])

y_predicted = y_predicted * mul_factor
y_test = y_test * mul_factor

#Final graph

st.subheader('Predictions vs Originals')

fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original')
plt.plot(y_predicted, 'r', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#Last prediction

st.subheader('Last datapoint prediction')
st.write(y_predicted[y_predicted.shape[0]-1][0])
