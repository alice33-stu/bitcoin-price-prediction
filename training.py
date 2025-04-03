import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import pandas as pd

# check GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# ascending order when reading data
data = pd.read_csv('bitcoin_price.csv', parse_dates=['timestamp']).sort_values(by='timestamp')
print(data.head())

data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# standardardize price
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

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
print(X[0])
print(Y[0])

# split train and test dataset 
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

# reshape [samples, time steps, features] for LSTM 
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(X_train[0])
print(Y_train[0])

# structure model 
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

print("Start Training")

# model training 
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=512)

model.save('lstm_bitcoin.h5')

# draw plot 
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.png')
plt.close()

# prediction 
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# inverse transform data 
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# calculate RMSE
import math
from sklearn.metrics import mean_squared_error

train_score = math.sqrt(mean_squared_error(Y_train, train_predict[:,0]))
test_score = math.sqrt(mean_squared_error(Y_test, test_predict[:,0]))

print(f'Train Score: {train_score:.2f} RMSE')
print(f'Test Score: {test_score:.2f} RMSE')

# compare the predicted and acutal results 
# plot
plt.figure(figsize=(14, 5))
plt.plot(data['close'], label='Actual Price')
plt.plot(pd.DataFrame(train_predict, index=data[:train_size].index), label='Train Predict')
plt.plot(pd.DataFrame(test_predict, index=data[train_size+time_step:].index), label='Test Predict')
plt.legend()
plt.savefig('actual_vs_predicted.png')
plt.close()

import json
with open("training_history.json", "w") as f:
    json.dump(history.history, f)
