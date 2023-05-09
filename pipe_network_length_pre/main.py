import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

data = pd.read_excel('Homework3.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data['value'] = np.log(data['value'].values)
data.set_index('Date', inplace=True)

plt.plot(data)
plt.title('Raw Data')
plt.xlabel('time')
plt.ylabel('Value')
plt.savefig('raw_data.png')
plt.close()

# smooth
value = data['value'].values
smoothed = lowess(value, range(len(value)), frac=0.3)
smoothed_df = pd.DataFrame({'value': smoothed[:, 1]}, index=data.index)

plt.plot(smoothed_df, label='Smoothed Data')
plt.title('Smoothed Data')
plt.xlabel('time')
plt.ylabel('Value')
plt.legend()
plt.savefig('smooth_data.png')
plt.close()

# ADF
diff = smoothed_df.diff(12).dropna()
result = sm.tsa.stattools.adfuller(diff)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# seasonal decompose
seasonal_result = sm.tsa.seasonal_decompose(data, model='additive')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,8))
seasonal_result.observed.plot(ax=ax1)
ax1.set_ylabel('Observed')
seasonal_result.trend.plot(ax=ax2)
ax2.set_ylabel('Trend')
seasonal_result.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonal')
seasonal_result.resid.plot(ax=ax4)
ax4.set_ylabel('Residual')
plt.savefig('seasonal_decompose.png')
plt.close()


fig, ax = plt.subplots(2, 1, figsize=(8,6))
sm.graphics.tsa.plot_acf(data, lags=30, ax=ax[0])
sm.graphics.tsa.plot_pacf(data, lags=30, ax=ax[1])
plt.savefig('ACF_PACF')  
plt.close()

p = 1
d = 1
q = 1

model = sm.tsa.ARIMA(data, order=(p, d, q))
results = model.fit()

n_periods = 6
forecast = results.predict(start=len(data), end=len(data)+n_periods-1)
data['value'] = np.exp(data['value'])
forecast = np.exp(forecast)
print(forecast)
smoothed_df['value'] = np.exp(smoothed_df['value'])
plt.plot(data, label='Original')
plt.plot(forecast.index, forecast, label='Forecast')
plt.plot(smoothed_df, label='Smoothed Data')
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Value')
plt.savefig('forcast_result.png')
plt.close()