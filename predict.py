import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# load data
data = pd.read_csv('bitcoin_price.csv', parse_dates=['timestamp']).sort_values(by='timestamp')
data['Date'] = pd.to_datetime(data['timestamp'])
data.set_index('Date', inplace=True)
data = data[['close']]  # use correct column name 'close'
data = data.sort_index()  # ascending order 

# standardardize data 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# create dataset 
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
X, Y = create_dataset(scaled_data, time_step)

# split train and test dataset 
train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# load model 
model = load_model('lstm_bitcoin.h5')

# do prediction 
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# inverse transform the data to the original range 
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
Y_train_inv = scaler.inverse_transform([Y_train])
Y_test_inv = scaler.inverse_transform([Y_test])

# plot the comparsion graph 
train_index = data.index[:len(Y_train)]
test_index = data.index[len(Y_train)+time_step:len(Y_train)+time_step+len(Y_test)]

plt.figure(figsize=(14, 6))
plt.plot(train_index, Y_train_inv[0], label='Actual Train Price')
plt.plot(test_index, Y_test_inv[0], label='Actual Test Price')
plt.plot(train_index, train_predict, label='Predicted Train Price')
plt.plot(test_index, test_predict, label='Predicted Test Price')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig('actual_vs_predicted.png')
plt.close()
