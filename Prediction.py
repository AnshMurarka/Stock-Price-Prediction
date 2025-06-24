import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv()  #Input link
df.head()

df['Date '] = pd.to_datetime(df['Date '])
df.head()

df.describe()
# Convert all columns except 'Date' and 'series' to numeric
cols_to_convert = ['OPEN ', 'HIGH ', 'LOW ', 'PREV. CLOSE ', 'ltp ','close ','vwap ', '52W H ', '52W L ', 'VOLUME ', 'VALUE ', 'No of trades ']

for col in cols_to_convert:
    df[col] = df[col].str.replace(r'[^\d.]', '', regex=True)  # Remove non-numeric characters
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float, replace errors with NaN

# Drop rows with NaN values if needed
df.dropna(inplace=True)  

# Verify conversion
print(df.dtypes)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Date '], df['close '], label='Closing Price', color='b')

# Format x-axis for better readability
plt.xticks(df['Date '][::100], rotation=45)
# Labels & Title
plt.xlabel('Date', fontsize=14)
plt.ylabel('Closing Price', fontsize=14)
plt.title('Stock Closing Prices Over Time', fontsize=16)
plt.legend()

# Show the plot
plt.show()
df1=df.reset_index()['close ']
##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size],df1[training_size:len(df1)]

training_size,test_size

# Using standard average technique
from sklearn.metrics import mean_squared_error
# SMA over a period of 10 and 20 years 
df['SMA_10'] = df['close '].rolling(10, min_periods=1).mean()
df['SMA_20'] = df['close '].rolling(20, min_periods=1).mean()
# Plot Closing Price, 10-year SMA, and 20-year SMA
plt.figure(figsize=(12,6))
plt.plot(df.index, df['close '], color='green', linewidth=3, label='Closing price')
plt.plot(df.index, df['SMA_10'], color='red', linewidth=3, label='10 days SMA')
plt.plot(df.index, df['SMA_20'], color='orange', linewidth=3, label='20 days SMA')
# Formatting
plt.xticks(fontsize=14)
plt.xlabel('Day', fontsize=16)
plt.ylabel('Closing price', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
#Calculate RMSE for SMA(10)
rmse_10 = np.sqrt(mean_squared_error(df['close '], df['SMA_10']))
# Calculate RMSE for SMA(20)
rmse_20 = np.sqrt(mean_squared_error(df['close '], df['SMA_20']))
print(f"RMSE for SMA(10): {rmse_10:.4f}")
print(f"RMSE for SMA(20): {rmse_20:.4f}")

# Exponential moving averages
# EMA Closing price
# Let's smoothing factor - 0.1
df['EMA_0.1'] = df['close '].ewm(alpha=0.1, adjust=False).mean()
# Let's smoothing factor  - 0.3
df['EMA_0.3'] = df['close '].ewm(alpha=0.3, adjust=False).mean()
# Let's smoothing factor  - 0.2
df['EMA_0.2'] = df['close '].ewm(alpha=0.2, adjust=False).mean()
# green - Closing price, red- smoothing factor - 0.1, yellow - smoothing factor  - 0.3, blue- smoothing 0.2
plt.figure(figsize=(12,6))
plt.plot(df.index, df['close '], color='green', linewidth=3, label='Closing price')
plt.plot(df.index, df['EMA_0.1'], color='red', linewidth=3, label='EMA-alpha=0.1')
plt.plot(df.index, df['EMA_0.3'], color='orange', linewidth=3, label='EMA-alpha=0.3')
plt.plot(df.index, df['EMA_0.2'], color='blue', linewidth=3, label='EMA-alpha=0.2')
# Formatting
plt.xticks(fontsize=14)
plt.xlabel('Day', fontsize=16)
plt.ylabel('Closing price', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
#Calculate RMSE for EMA(alpha=0.1)
rmse_1 = np.sqrt(mean_squared_error(df['close '], df['EMA_0.1']))
# Calculate RMSE for EMA(alpha=0.3)
rmse_3 = np.sqrt(mean_squared_error(df['close '], df['EMA_0.3']))
# Calculate RMSE for EMA(alpha=0.2)
rmse_2 = np.sqrt(mean_squared_error(df['close '], df['EMA_0.2']))
print(f"RMSE for EMA-alpha=0.1: {rmse_1:.4f}")
print(f"RMSE for EMA-alpha=0.2: {rmse_2:.4f}")
print(f"RMSE for EMA-alpha=0.3: {rmse_3:.4f}")

# LSTM
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
#splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size],df1[training_size:len(df1)]
#convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)
# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
print(X_train.shape), print(y_train.shape)
print(X_test.shape), print(ytest.shape)
# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
model=Sequential()
model.add(LSTM(100,return_sequences=True,input_shape=(100,1),activation='relu'))
Dropout(0.2)
#model.add(LSTM(100,return_sequences=True,activation='relu'))
#Dropout(0.2)
model.add(LSTM(100,activation='relu'))
Dropout(0.2)
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')
# Initialize EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=10,         # Stop after 10 epochs if no improvement
    restore_best_weights=True,  # Restore the best model weights
    verbose=1
)
model_history=model.fit(X_train, y_train,
                        epochs=100, validation_split=0.2, callbacks=[early_stopping])
import tensorflow as tf
tf.__version__
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
y_train=scaler.inverse_transform(y_train.reshape(-1,1))
ytest=scaler.inverse_transform(ytest.reshape(-1,1))
### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))
### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

## Classical Time series modelling
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Defining ADF Test to test the stationarity of the data
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] < 0.05:
        print("The data is stationary.")
    else:
        print("The data is NOT stationary.")

adf_test(df['close '])
df['close_Diff'] = df['close '].diff() # Taking the difference of closing price

# Plotting the difference of closing price across the data 
plt.figure(figsize=(12, 6))
plt.plot(df['close_Diff'], label="Differenced Bitcoin Price", color="red")
plt.title("Stock Price (First-Order Differencing)")
plt.legend()
plt.show()

# Check stationarity again
adf_test(df['close_Diff'].dropna())
# Select only 'close' and 'Date' columns
df_close = df['close ']

train, test = df_close[:int(len(df_close)*0.9)], df_close[int(len(df_close)*0.9):]

# Using simple ARIMA with differenciated series
# Auto ARIMA to find best (p,d,q)
auto_arima_model = pm.auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
print(f"Best ARIMA Order: {auto_arima_model.order}")
warnings.filterwarnings("ignore")
# Fit ARIMA model with optimal parameters
arima_model = ARIMA(train, order=auto_arima_model.order)
arima_fit = arima_model.fit()
forecast = arima_fit.get_forecast(steps=len(test))
conf = forecast.conf_int(alpha=0.05)
fc = forecast.predicted_mean
#Convert to pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf.iloc[:, 0], index=test.index)
upper_series = pd.Series(conf.iloc[:, 1], index=test.index)
#Plot
plt.figure(figsize=(10,5), dpi=100)
plt.plot(train, label='Training Data')
plt.plot(test, color='blue', label='Actual Stock Price')
plt.plot(fc_series, color='orange', label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
plt.title('HDFC BANK Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('HDFC BANK Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()
#Report Performance
mse = mean_squared_error(test, fc_series)
mae = mean_absolute_error(test, fc_series)
rmse = math.sqrt(mse)
mape = np.mean(np.abs(fc_series - test) / np.abs(test))
print('MSE:', mse)
print('MAE:', mae)
print('RMSE:', rmse)
print('MAPE:', mape)

#Using ARIMA with log transformation
train_log = np.log(train)
test_log = np.log(test)
# Fit ARIMA on log-transformed data
auto_arima_model = pm.auto_arima(train_log, seasonal=False, stepwise=True, suppress_warnings=True)
arima_model = ARIMA(train_log, order=auto_arima_model.order)
arima_fit = arima_model.fit()
forecast_log = arima_fit.get_forecast(steps=len(test_log))
fc_log = forecast_log.predicted_mean
conf_log = forecast_log.conf_int()
# Convert back from log
fc = np.exp(fc_log)
conf = np.exp(conf_log)
#Convert to pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf.iloc[:, 0], index=test.index)
upper_series = pd.Series(conf.iloc[:, 1], index=test.index)
#Plot
plt.figure(figsize=(10,5), dpi=100)
plt.plot(train, label='Training Data')
plt.plot(test, color='blue', label='Actual Stock Price')
plt.plot(fc_series, color='orange', label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
plt.title('HDFC BANK Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('HDFC BANK Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()
#Report Performance
mse = mean_squared_error(test, fc_series)
mae = mean_absolute_error(test, fc_series)
rmse = math.sqrt(mse)
mape = np.mean(np.abs(fc_series - test) / np.abs(test))
print('MSE:', mse)
print('MAE:', mae)
print('RMSE:', rmse)
print('MAPE:', mape)

# Try a few variants
orders_to_try = [(1,1,1), (2,0,2), (3,1,3), (4,1,2)]
for order in orders_to_try:
    model = ARIMA(train_log, order=order)
    results = model.fit()
    pred_log = results.forecast(steps=len(test_log))
    pred = np.exp(pred_log)
    rmse = math.sqrt(mean_squared_error(test, pred))
    print(f"Order {order} -> RMSE: {rmse:.2f}")

# SARIMAX with log transformation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm
import math
# Assume your df already contains 'Date' and 'close '
df['close'] = df['close ']  # Remove extra space in column name if needed
# Calculate EMA and MACD
df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA_12"] - df["EMA_26"]
# Drop initial NaNs
df.dropna(inplace=True)
# Create log-transformed target
y = np.log(df["close"])
# Define exogenous features
X = df[["EMA_12", "EMA_26", "MACD"]]
# Train-test split
train_size = int(len(df) * 0.9)
train, test = y[:train_size], y[train_size:]
exog_train, exog_test = X[:train_size], X[train_size:]
# Find best ARIMA order
auto_model = pm.auto_arima(train, exogenous=exog_train, seasonal=False, stepwise=True, suppress_warnings=True)
p, d, q = auto_model.order
print(f"Best ARIMA Order: {(p, d, q)}")
# Fit ARIMAX model
model = ARIMA(endog=train, exog=exog_train, order=(p, d, q))
fit = model.fit()
# Forecast
forecast_log = fit.forecast(steps=len(test), exog=exog_test)
forecast = np.exp(forecast_log)  # convert back from log
# Actual values
actual = np.exp(test)
# Evaluation
rmse = np.sqrt(mean_squared_error(actual, forecast))
mae = mean_absolute_error(actual, forecast)
mape = np.mean(np.abs(forecast - actual) / np.abs(actual)) * 100
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape)
# Plot
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(np.exp(train), label='Training Data')
plt.plot(actual, color='blue', label='Actual Stock Price')
plt.plot(forecast, color='orange', label='Predicted Stock Price')
plt.title('HDFC BANK Stock Price Prediction using ARIMAX with MACD')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# ACF, PACF Plot
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Example: If you have a CSV or DataFrame column
# data = pd.read_csv("your_data.csv")["closing_price"]
# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(df['close '], lags=40, ax=plt.gca(), title="Autocorrelation (ACF)")
plt.subplot(1, 2, 2)
plot_pacf(df['close '], lags=40, ax=plt.gca(), title="Partial Autocorrelation (PACF)")
plt.tight_layout()
plt.show()

# SARIMAX without log transformation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm
import math
# Assume your df already contains 'Date' and 'close '
df['close'] = df['close ']  # Remove extra space in column name if needed
# Calculate EMA and MACD
df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["Signal"] =df["MACD"].ewm(span=9, adjust=False).mean()
def calculate_rsi(data, window=14):
    delta = data['close '].diff()  # price changes
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data
df=calculate_rsi(df)
# Drop initial NaNs
df.dropna(inplace=True)
# Create log-transformed target
y = df["close"]
# Define exogenous features
X = df[["EMA_12","EMA_26","RSI"]]
# Train-test split
train_size = int(len(df) * 0.9)
train, test = y[:train_size], y[train_size:]
exog_train, exog_test = X[:train_size], X[train_size:]
# Find best ARIMA order
auto_model = pm.auto_arima(train, exogenous=exog_train, seasonal=False, stepwise=True, suppress_warnings=True)
p, d, q = auto_model.order
print(f"Best ARIMA Order: {(p, d, q)}")
# Fit ARIMAX model
model = ARIMA(endog=train, exog=exog_train, order=(p, d, q))
fit = model.fit()
# Forecast
forecast = fit.forecast(steps=len(test), exog=exog_test)
# Actual values
actual = test
# Evaluation
rmse = np.sqrt(mean_squared_error(actual, forecast))
mae = mean_absolute_error(actual, forecast)
mape = np.mean(np.abs(forecast - actual) / np.abs(actual)) * 100
print("RMSE:", rmse)
# Plot
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, actual, color='blue', label='Actual Stock Price')
plt.plot(test.index, forecast, color='orange', label='Predicted Stock Price')
plt.title('HDFC BANK Stock Price Prediction using ARIMAX with EMA_12, EMA_26, RSI')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# SARIMAX with RSI, MACD features only
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm
import math
# Assume your df already contains 'Date' and 'close '
df['close'] = df['close ']  # Remove extra space in column name if needed
# Calculate EMA and MACD
df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["Signal"] =df["MACD"].ewm(span=9, adjust=False).mean()
def calculate_rsi(data, window=14):
    delta = data['close '].diff()  # price changes
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data
df=calculate_rsi(df)
# Drop initial NaNs
df.dropna(inplace=True)
# Create log-transformed target
y = df["close"]
# Define exogenous features
X = df[["MACD","RSI"]]
# Train-test split
train_size = int(len(df) * 0.9)
train, test = y[:train_size], y[train_size:]
exog_train, exog_test = X[:train_size], X[train_size:]
# Find best ARIMA order
auto_model = pm.auto_arima(train, exogenous=exog_train, seasonal=False, stepwise=True, suppress_warnings=True)
p, d, q = auto_model.order
print(f"Best ARIMA Order: {(p, d, q)}")
# Fit ARIMAX model
model = ARIMA(endog=train, exog=exog_train, order=(p, d, q))
fit = model.fit()
# Forecast
forecast = fit.forecast(steps=len(test), exog=exog_test)
# Actual values
actual = test
# Evaluation
rmse = np.sqrt(mean_squared_error(actual, forecast))
mae = mean_absolute_error(actual, forecast)
mape = np.mean(np.abs(forecast - actual) / np.abs(actual)) * 100
print("RMSE:", rmse)
# Plot
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, actual, color='blue', label='Actual Stock Price')
plt.plot(test.index, forecast, color='orange', label='Predicted Stock Price')
plt.title('HDFC BANK Stock Price Prediction using ARIMAX with MACD, RSI')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
