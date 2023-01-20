import os
from datetime import timedelta

from flask import Flask, request, render_template
from keras.models import load_model
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from sklearn.preprocessing import MinMaxScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('MyModel_stockpred.h5')
scaler = MinMaxScaler(feature_range=(0, 1))


# Define a function to generate the dataset
def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the user input
        symbol = request.form['symbol']
        # Download the stock prices for the specified symbol
        stock_data = yf.download(symbol)
        # Generate future dates
        last_date = stock_data['Close'].index[-1]
        future_dates = []
        for i in range(30):
            future_dates.append((last_date + timedelta(days=i + 1)).strftime('%m/%d'))

        # Extract the stock price column
        stock_prices = stock_data['Close'].values.reshape(-1, 1)

        # Scale the stock prices to be between 0 and 1
        scaled_stock_prices = scaler.fit_transform(stock_prices)

        # Generate the dataset with a lookback of 60 days
        look_back = 60
        trainX, trainY = create_dataset(scaled_stock_prices, look_back)

        # Reshape the dataset for use with the LSTM model
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

        # Make predictions
        predictions = model.predict(trainX)

        # Reverse the scaling on the predictions
        predictions = scaler.inverse_transform(predictions)

        # Plot the predictions and actual stock prices
        #plt.plot(stock_prices, label='Actual')
        plt.plot(future_dates[:30],predictions[:30], label='Predicted')

        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.1)
        #plt.tick_params(axis='x', which='major', pad=15)
        plt.tick_params(axis='x', labelsize=8)
        #plt.gca().xaxis.set_major_locator(future_dates.AutoDateLocator(minticks=3, maxticks=10))
        plt.legend()


        # Render the plot as a PNG image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
        buf.close()
        plt.clf()

        return render_template('index.html', image_base64=image_base64)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
