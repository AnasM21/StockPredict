import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load the stock price dataset
df = pd.read_csv('stock_prices.csv')

# Extract the stock price column
stock_prices = df['Close'].values.reshape(-1, 1)

# Scale the stock prices to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_stock_prices = scaler.fit_transform(stock_prices)

# Split the dataset into training and testing sets
train_size = int(len(scaled_stock_prices) * 0.8)
test_size = len(scaled_stock_prices) - train_size
train_data, test_data = scaled_stock_prices[0:train_size,:], scaled_stock_prices[train_size:len(scaled_stock_prices),:]

# Create a function to generate the training and testing datasets
def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Set the number of time steps (days) to look back
look_back = 60

# Generate the training and testing datasets
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)

# Reshape the training and testing datasets for use with the LSTM model
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(LSTM(units=100))
model.add(Dense(1))

# Compile and fit the model
model.compile(loss='mae', optimizer='adam')
history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), shuffle=False)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Make predictions on the test data
predictions = model.predict(testX)


#Reverse the scaling on the predictions and the actual test data
predictions = scaler.inverse_transform(predictions)
testY = scaler.inverse_transform(testY.reshape(-1, 1))

#Calculate the root mean squared error of the predictions
rmse = np.sqrt(np.mean((predictions - testY)**2))
print('Root Mean Squared Error:', rmse)

#Plot the predictions and the actual data
plt.plot(testY, label='actual')
plt.plot(predictions, label='predicted')
plt.legend()
plt.show()

model.save('MyModel_stockpred.h5')

