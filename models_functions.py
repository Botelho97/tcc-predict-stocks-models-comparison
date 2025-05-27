import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt

rcParams["figure.figsize"] = 18, 6

def custom_train_test(series: pd.Series, datepoint: str = "2024-07") -> list[pd.Series]:
    y_train = series[series.index < datepoint]
    y_test = series[series.index >= datepoint]
    return y_train, y_test

def get_metrics(y_test:pd.Series, forecast:pd.Series) -> list[float]:
    mape = mean_absolute_percentage_error(y_test, forecast)
    rmse = root_mean_squared_error(y_test, forecast)
    r2 = r2_score(y_test, forecast)
    return mape, rmse, r2

def naive_forecast(y_train: pd.Series, y_test: pd.Series) -> list:
    model_name = "Naive"
    forecast = pd.Series(y_train.iloc[-1] * len(y_test), index=y_test.index)
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return model_name, mape, rmse, r2, forecast


def mean_forecast(y_train: pd.Series, y_test: pd.Series) -> list:
    model_name = "Mean"
    forecast = pd.Series(y_train.mean(), index=y_test.index)
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return model_name, mape, rmse, r2, forecast


def drift_forecast(y_train: pd.Series, y_test: pd.Series) -> list:
    model_name = "Drift"
    n = len(y_train)
    drift_slope = (y_train.iloc[-1] - y_train.iloc[0]) / (n - 1)
    forecast = y_train.iloc[-1] + drift_slope * np.arange(1, len(y_test) + 1)
    forecast = pd.Series(forecast, index=y_test.index)
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return model_name, mape, rmse, r2, forecast


def naive_seasonal_forecast(y_train:pd.Series, y_test:pd.Series, seasonal: int = 5) -> list:
    model_name = "Naive Seasonal"
    forecast = pd.Series([y_train.iloc[-seasonal + i % seasonal] for i in range(len(y_test))], index=y_test.index)
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return model_name, mape, rmse, r2, forecast


def ses_forecast(y_train: pd.Series, y_test: pd.Series) -> list:
    pass


def ts_pipeline(y_train: pd.Series, y_test: pd.Series):
    pass

