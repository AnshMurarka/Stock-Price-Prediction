# Stock-Price-Prediction
Model summary:
-> SMA
RMSE 10 day 36.1325
RMSE 20 days 49.5691
-> EMA
Alpha (0.1) RMSE 40.7280 (Usually, alpha=2/(span+1))
Alpha (0.2) RMSE 28.0648 (commonly used)
Alpha (0.3) RMSE 21.0648
-> For LSTM: 
100 -> dropout (0.2) -> 100 -> dropout (0.2) -> 100 -> dropout (0.2) -> Dense (1)
For 100 steps using Sigmoid function,
Train RMSE = 21.762
Test RMSE = 22.9357
For 100 steps using ReLU achieved 24.15 RMSE 

MACD-> Momentum indicator
EMA_12 and EMA_26-> Long- and short-term trends
Volume-> Measures market activity
RSI-> Overbought or oversold signal (not used here)
