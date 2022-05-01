# Cryptocurrency Buy Signal Prediction with Deep Neural Network – Q Learning

This simple TensorFlow app try to learn what is the best time to buy Crypto Asset. For this goal using the Bitcoin historical data (5 minute chart) and validate the trained model with the Ethereum historical data.
Basic concept is create a simple model and provide all the information for it. This model input parameters are the candle’s data and technical indicators. More over the model get separate signals when the indicators give a buy signal. 

## Used Indicators:
-	**EMA** – 200 period of Exponential Moving Average, buy signal when the price is above the EMA line.
-	**MACD** – MACD line and Signal line, buy signal MACD cross (all MACD cross not filtered out the above 0 crosses )
-	**CMF** – Chaikin Money Flow indicator, buy signal when the indicator is above 0 
-	**ATR** – Average True Rage – determinate the Stop Loss level and Take Profit level.
-	**RSI** – Relative Strength Index, buy signal when the indicator shows oversold (below 30), extra buy signal when RSI left the 30 level to up.
-	**ADX** – Trend Strength Indicator (**14 candles**), buy signal when the indicator is above 50
-	**ADX** – Trend Strength Indicator (**288 candles**), buy signal when the indicator is above 50
-	**SMA** – Simple Moving Average 1440 periods (5 days) buy signal when the price is above the SMA line
-	**STOCHASTIC K** line and **STOCHASTIC D** line buy signal when stochastic below 20 extra buy signal when K line cross D line below 20 

## Hyperparameters 
-	**ACTIVATION**: default ‘Sigmoid’ activation function for the keras model. 
-	**GAMMA**: default 0.99 set how important is the current reward
-	**LR**: default 0.00000025 learning rate shows how important the current values.
-	**EPSILON_DEC**: default 1e-5 epsilon decay value
-	**EPSILON_START** default 1 random step /predicted step rate start value
-	**EPSILON_END** default 0.1 random step /predicted step rate minimum value

## Parameters:
-	**N_TRADE**: number of trade. The app stops after this amount of trade
-	**RENDER_MODE**: 'compueter' or 'human' show or hide the chart data. *Human mode not implemented* *
-	**CANDLES**: default 72 set up how many candles are the input parameter.

## Monitor to learning progress use the TensorBoard 
Run the following command from the source folder 
```
tensorboard --logdir './logs/train' 
```

## DISCLAIMER
**I am not a financial advisor.** This code is for learning purpose only. Trading with crypto assets is extremely risky. **The value of your investment can go down as well as up, and you may get back less than you invest.**

