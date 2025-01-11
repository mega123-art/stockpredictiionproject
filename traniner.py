import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
end=datetime.now()
start=datetime(end.year-20,end.month,end.day)
stock="GOOG"
google_data = yf.download(stock, start, end)
#data analysis
print(google_data.head())
google_data.shape()
google_data.describe()
google_data.info()
#no null values
google_data.isna().sum()
plt.figure(figsize = (15,5))
google_data['Adj Close'].plot()
plt.xlabel("years")
plt.ylabel("Adj Close")
plt.title("Closing price of Google data")
plt.show()
def plot_graph(figsize, values, column_name):
    plt.figure()
    values.plot(figsize = figsize)
    plt.xlabel("years")
    plt.ylabel(column_name)
    plt.title(f"{column_name} of Google data")

google_data.columns()
for column in google_data.columns:
  plot_graph((15,5),google_data[column],column)

#using moving average checking for different number of days
#for 250 days
google_data['MA_for_250_days'] = google_data['Adj Close'].rolling(250).mean()
plot_graph((15,5), google_data['MA_for_250_days'], 'MA_for_250_days')
plot_graph((15,5), google_data[['Adj Close','MA_for_250_days']], 'MA_for_250_days')
#for 100 days
google_data['MA_for_100_days'] = google_data['Adj Close'].rolling(100).mean()
plot_graph((15,5), google_data[['Adj Close','MA_for_100_days']], 'MA_for_100_days')

plot_graph((15,5), google_data[['Adj Close','MA_for_100_days', 'MA_for_250_days']], 'MA')
google_data['percentage_change_cp'] = google_data['Adj Close'].pct_change()
google_data[['Adj Close','percentage_change_cp']].head()
plot_graph((15,5), google_data['percentage_change_cp'], 'percentage_change')

Adj_close_price = google_data[['Adj Close']]
max(Adj_close_price.values),min(Adj_close_price.values) 
#scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(Adj_close_price)
scaled_data
len(scaled_data)

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])
    

x_data, y_data = np.array(x_data), np.array(y_data)

#splitting the data
splitting_len = int(len(x_data)*0.7)
x_train = x_data[:splitting_len]
y_train = y_data[:splitting_len]

x_test = x_data[splitting_len:]
y_test = y_data[splitting_len:]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(64,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs = 2)
model.summary()

predictions = model.predict(x_test)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_test)
rmse = np.sqrt(np.mean( (inv_predictions - inv_y_test)**2))


ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_predictions.reshape(-1)
 } ,
    index = google_data.index[splitting_len+100:]
)
ploting_data.head()

plot_graph((15,6), ploting_data, 'test data')
plot_graph((15,6), pd.concat([Adj_close_price[:splitting_len+100],ploting_data], axis=0), 'whole data')
model.save("Latest_stock_price_model.keras")
