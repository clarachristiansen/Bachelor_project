import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = pd.read_parquet("Data/DSB_BDK_trainingset.parquet")
togpunktlighed_daily = data.groupby('dato')['togpunktlighed'].mean()
togpunktlighed_daily = pd.DataFrame(togpunktlighed_daily).reset_index()

ratio = 0.9
train, test = togpunktlighed_daily[:int(togpunktlighed_daily.shape[0]*ratio)], togpunktlighed_daily[int(togpunktlighed_daily.shape[0]*ratio):]

SARIMA_model = auto_arima(train['togpunktlighed'], 
                        start_p=1, start_q=1,
                        test='adf',
                        max_p=3, max_q=3, 
                        max_P=3, max_Q=3,
                        m=365, #12 is the frequncy of the cycle
                        start_P=0, 
                        seasonal=True, #set to seasonal
                        d=1, # None if done via test 
                        D=1, #order of the seasonal differencing
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True,
                        n_jobs=-1)

n_periods=test.shape[0]
fitted, confint = SARIMA_model.predict(n_periods=n_periods, return_conf_int=True)

mse = mean_squared_error(fitted, test)

plt.figure(figsize=(15,7))
plt.plot(togpunktlighed_daily['togpunktlighed'][-n_periods:], color='#1f76b4')
plt.plot(fitted, color='darkgreen')

plt.title("ARIMA/SARIMA - Forecast of Airline Passengers")
plt.show()