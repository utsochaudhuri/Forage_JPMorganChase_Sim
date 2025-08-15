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

def get_natgas_price(date_str):
    date = pd.to_datetime(date_str)
    if date <= df.index[-1]:
        val = float(f_linear(date.toordinal()))
        return val
    else:
        months_ahead = (date.year - df.index[-1].year) * 12 + (date.month - df.index[-1].month)
        months_ahead = max(1, months_ahead)  # Ensure at least 1 month ahead
        fc = stepwise_model.predict(n_periods=months_ahead, return_conf_int=True)
        val = float(fc[-1])
        return val

def month_diff(date1, date2):
    """Calculate the difference in months between two dates."""
    return (date2.year - date1.year) * 12 + (date2.month - date1.month)

# Pricing function
def pricing_func(inject_date, with_date, inject_with_price, max_vol, storage_price, transport_price):
    """    Calculate the profit from injecting and withdrawing natural gas."""
    inject_price = get_natgas_price(inject_date)
    with_price = get_natgas_price(with_date)
    # Storage cost calculation (storage cost is monthly)
    storage_cost = storage_price * month_diff(pd.to_datetime(inject_date), pd.to_datetime(with_date))
    # Injection/Withdrawal cost (cost per 1 million MMBTU)
    inject_with_cost = (max_vol/1000000) * inject_with_price
    # Transport cost (single cost for the entire volume)
    transport_cost = transport_price * 2

    # Profit calculation
    profit = (with_price - inject_price * 1000000) - storage_cost - inject_with_cost - transport_cost
    return profit

# Example usage
print(pricing_func("2023-05-15", "2024-01-15", 10000, 10000000, 2000, 5000))