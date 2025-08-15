import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pmdarima import auto_arima

file_path = 'Task 1/Nat_Gas.csv'
df = pd.read_csv(file_path, index_col='Dates', parse_dates=True).sort_index()

dates_num = df.index.map(pd.Timestamp.toordinal).to_numpy()
prices = df['Prices'].to_numpy()

f_linear = interp1d(dates_num, prices, kind='linear')

stepwise_model = auto_arima(
    df['Prices'],
    seasonal=True, m=12,
    trace=False,
    suppress_warnings=True
)

n_periods_default = 12
forecast_default, conf_int_default = stepwise_model.predict(n_periods=n_periods_default, return_conf_int=True)
future_dates_default = pd.date_range(df.index[-1] + pd.offsets.MonthEnd(), periods=n_periods_default, freq='M')
forecast_df_default = pd.DataFrame(
    {'Forecast': forecast_default, 'Lower CI': conf_int_default[:, 0], 'Upper CI': conf_int_default[:, 1]},
    index=future_dates_default
)

plt.figure(figsize=(10, 5))
plt.plot(df['Prices'], label='History')
plt.plot(forecast_df_default['Forecast'], label='12M Forecast')
plt.fill_between(future_dates_default, forecast_df_default['Lower CI'], forecast_df_default['Upper CI'], alpha=0.3)
plt.legend()
plt.title('Natural Gas: History + 12M Forecast')
plt.xlabel('Date'); plt.ylabel('Price')
plt.grid(True, alpha=0.3)
plt.show()

def get_natgas_price(date_str, plot=True):
    date = pd.to_datetime(date_str)
    if date <= df.index[-1]:
        val = float(f_linear(date.toordinal()))
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, prices, 'o', label='Data Points')
            plt.plot(df.index, f_linear(dates_num), '-', label='Linear Interpolation')
            plt.axvline(date, linestyle='--', label=f'Query: {date.date()}')
            plt.title('Back-Estimate via Interpolation')
            plt.xlabel('Date'); plt.ylabel('Price')
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.show()
        return val
    else:
        months_ahead = (date.year - df.index[-1].year) * 12 + (date.month - df.index[-1].month)
        fc, ci = stepwise_model.predict(n_periods=months_ahead, return_conf_int=True)
        val = float(fc[-1])
        if plot:
            future_idx = pd.date_range(df.index[-1] + pd.offsets.MonthEnd(), periods=months_ahead, freq='M')
            plt.figure(figsize=(10, 5))
            plt.plot(df['Prices'], label='History')
            plt.plot(future_idx, fc, label=f'Forecast to {future_idx[-1].date()}')
            plt.fill_between(future_idx, ci[:, 0], ci[:, 1], alpha=0.3)
            plt.axvline(future_idx[-1], linestyle='--', label=f'Query: {date.date()}')
            plt.title('Forecast via SARIMA')
            plt.xlabel('Date'); plt.ylabel('Price')
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.show()
        return val

# Example usage:
print(get_natgas_price("2023-05-15", plot=True))
print(get_natgas_price("2025-03-15", plot=True))