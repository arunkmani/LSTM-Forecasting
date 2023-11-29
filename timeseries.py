import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras
from keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.python.keras.models import Sequential
seq_len=60
def seq_splitter(data):
    number_of_splits= len(data) - seq_len + 1
    return np.array([data[i:seq_len] for i in range(number_of_splits)])
def get_splits(data,fraction):
        sequences = seq_splitter(scaled_values)
        print(len(sequences))
        full_length =sequences.shape[0]
        n_train = int( full_length * fraction)
        x_train = sequences[:n_train]
        y_train = sequences[:n_train]
        x_test = sequences[n_train:]
        y_test = sequences[n_train:]
        return x_train, y_train, x_test, y_test

df=pd.read_csv('sensor1.csv')
plt.plot(df['Temp'])
plt.show()
scaler = MinMaxScaler()
values = df.Temp.values.reshape(-1,1)
scaled_values = scaler.fit_transform(values)
x_train, y_train, x_test, y_test = get_splits(scaled_values,.9)

dropout = 0.2
window_size = seq_len - 1
model = keras.Sequential()
model.add(LSTM(window_size, return_sequences=True,input_shape=(window_size, x_train.shape[-1])))
model.add(Dropout(rate=dropout))
model.add(Bidirectional(LSTM((window_size * 2), return_sequences=True))) 
model.add(Dropout(rate=dropout))
model.add(Bidirectional(LSTM(window_size, return_sequences=False))) 
model.add(Dense(units=1))
model.add(Activation('linear'))

batch_size = 16

model.compile(loss='mean_squared_error',optimizer='adam')

history = model.fit(x_train,y_train,epochs=10, batch_size=batch_size,shuffle=False,validation_split=0.2)

y_pred = model.predict(x_test)

# invert the scaler to get the absolute price data
y_test_orig = scaler.inverse_transform(y_test)
y_pred_orig = scaler.inverse_transform(y_pred)

# plots of prediction against actual data
plt.plot(y_test_orig, label='Actual Price', color='orange')
plt.plot(y_pred_orig, label='Predicted Price', color='green')
plt.ylabel('Price ($)')
plt.legend(loc='best')

plt.show()