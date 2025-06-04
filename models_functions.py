import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox  # Testar para ver se os resíduos são independentes
from scipy.stats import kstest  # Para testar se os resíduos seguem distribuição normal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Time Series Models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from pmdarima import auto_arima
# Pending Prophet

# ML Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# DL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def custom_train_test(data: pd.Series, datepoint: str = "2024-07"):
    data_train = data[data.index < datepoint]
    data_test = data[data.index >= datepoint]
    return data_train, data_test

def get_metrics(y_test: pd.Series, forecast: pd.Series) -> list[float]:
    mape = mean_absolute_percentage_error(y_test, forecast)
    rmse = root_mean_squared_error(y_test, forecast)
    r2 = r2_score(y_test, forecast)
    return mape, rmse, r2

def naive_forecast(y_train: pd.Series, y_test: pd.Series) -> dict:
    model_name = "Naive"
    forecast = pd.Series([y_train.iloc[-1]] * len(y_test), index=y_test.index)
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return {"Nome do Modelo": model_name,
            "Previsão": forecast,
            "MAPE": mape,
            "RMSE": rmse,
            "R2": r2,
            "Modelo": None}



def mean_forecast(y_train: pd.Series, y_test: pd.Series) -> list:
    model_name = "Mean"
    forecast = pd.Series(y_train.mean(), index=y_test.index)
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return {"Nome do Modelo": model_name,
            "Previsão": forecast,
            "MAPE": mape,
            "RMSE": rmse,
            "R2": r2,
            "Modelo": None}


def drift_forecast(y_train: pd.Series, y_test: pd.Series) -> list:
    model_name = "Drift"
    n = len(y_train)
    drift_slope = (y_train.iloc[-1] - y_train.iloc[0]) / (n - 1)
    forecast = y_train.iloc[-1] + drift_slope * np.arange(1, len(y_test) + 1)
    forecast = pd.Series(forecast, index=y_test.index)
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return {"Nome do Modelo": model_name,
            "Previsão": forecast,
            "MAPE": mape,
            "RMSE": rmse,
            "R2": r2,
            "Modelo": None}


def naive_seasonal_forecast(y_train: pd.Series, y_test: pd.Series, seasonal_period: int = 5) -> list:
    model_name = "Naive Seasonal"
    forecast = pd.Series([y_train.iloc[-seasonal_period + i % seasonal_period] for i in range(len(y_test))], index=y_test.index)
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return {"Nome do Modelo": model_name,
            "Previsão": forecast,
            "MAPE": mape,
            "RMSE": rmse,
            "R2": r2,
            "Modelo": None}


def ses_forecast(y_train: pd.Series, y_test: pd.Series) -> list:
    model_name = "SES"
    ses_model = SimpleExpSmoothing(y_train).fit()
    forecast = ses_model.forecast(steps=len(y_test))
    forecast.index = y_test.index
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return {"Nome do Modelo": model_name,
            "Previsão": forecast,
            "MAPE": mape,
            "RMSE": rmse,
            "R2": r2,
            "Modelo": ses_model}


def holt_forecast(y_train: pd.Series, y_test: pd.Series) -> list:
    model_name = "Holt"
    holt_model = Holt(y_train).fit()
    forecast = holt_model.forecast(steps=len(y_test))
    forecast.index = y_test.index
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return {"Nome do Modelo": model_name,
            "Previsão": forecast,
            "MAPE": mape,
            "RMSE": rmse,
            "R2": r2,
            "Modelo": holt_model}


def holt_winters_add_forecast(y_train: pd.Series, y_test: pd.Series, seasonal_period: int = 5) -> list:
    model_name = "Holt-Winters Aditivo"
    hw_add_model = ExponentialSmoothing(y_train, seasonal_periods=seasonal_period, trend='add', seasonal='add').fit()
    forecast = hw_add_model.forecast(steps=len(y_test))
    forecast.index = y_test.index
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return {"Nome do Modelo": model_name,
            "Previsão": forecast,
            "MAPE": mape,
            "RMSE": rmse,
            "R2": r2,
            "Modelo": hw_add_model}


def holt_winters_mult_forecast(y_train: pd.Series, y_test: pd.Series, seasonal_period: int = 5) -> list:
    model_name = "Holt-Winter Multiplicativo"
    hw_mult_model = ExponentialSmoothing(y_train, seasonal_periods=seasonal_period, trend='add', seasonal='mul').fit()
    forecast = hw_mult_model.forecast(steps=len(y_test))
    forecast.index = y_test.index
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return {"Nome do Modelo": model_name,
            "Previsão": forecast,
            "MAPE": mape,
            "RMSE": rmse,
            "R2": r2,
            "Modelo": hw_mult_model}


def arima_forecast(y_train: pd.Series, y_test: pd.Series) -> list:
    model_name = "ARIMA"
    auto_arima_model = auto_arima(y_train, trace=True, seasonal=False, stepwise=True)
    forecast = auto_arima_model.predict(len(y_test))
    forecast.index = y_test.index
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return {"Nome do Modelo": model_name,
            "Previsão": forecast,
            "MAPE": mape,
            "RMSE": rmse,
            "R2": r2,
            "Modelo": auto_arima_model}


def sarima_forecast(y_train:pd.Series, y_test:pd.Series, seasonal_period: int=5) -> list:
    model_name = "SARIMA"
    auto_sarima_model = auto_arima(y_train, trace=True, seasonal=True, m=seasonal_period, stepwise=True)
    forecast = auto_sarima_model.predict(len(y_test))
    forecast.index = y_test.index
    mape, rmse, r2 = get_metrics(y_test, forecast)
    return {"Nome do Modelo": model_name,
            "Previsão": forecast,
            "MAPE": mape,
            "RMSE": rmse,
            "R2": r2,
            "Modelo": auto_sarima_model}


def resid_tests(model) -> None:
    resid = model.resid()
    # Ljung-box
    ljungbox_result = acorr_ljungbox(resid, lags=30, return_df=True)
    p_value_ljungbox = ljungbox_result.lb_pvalue.values[0]
    print(f'p-valor ljung-box {p_value_ljungbox}')
    if p_value_ljungbox > 0.05:
        print(f'H0: p-valor > 0.05 -> Os resíduos são independentes (iid), o modelo está bem ajustado')
    else:
        print(f'H1: p-valor <= 0.05 -> Os resíduos não são independentes, o modelo possui falhas no ajuste.')
    # Kolmogorov-Smirnov
    ks_stat, p_value_ks = kstest(resid, 'norm', args=(np.mean(resid), np.std(resid)))
    print(f'Teste de Kolmogorov-Smirnov para normalidade: p-valor = {p_value_ks}')
    if p_value_ks > 0.01:
        print("Os resíduos seguem uma distribuição normal.")
    else:
        print("Os resíduos não seguem uma distribuição normal.")
    # Arch -> Implement later.


def ts_pipeline(series: pd.Series, seasonal=5) -> pd.DataFrame:
    y_train, y_test = custom_train_test(series)
    models = []
    naive = naive_forecast(y_train, y_test)
    models.append(naive)
    mean = mean_forecast(y_train, y_test)
    models.append(mean)
    drift = drift_forecast(y_train, y_test)
    models.append(drift)
    naive_seasonal = naive_seasonal_forecast(y_train, y_test, seasonal)
    models.append(naive_seasonal)
    ses = ses_forecast(y_train, y_test)
    models.append(ses)
    holt = holt_forecast(y_train, y_test)
    models.append(holt)
    hw_add = holt_winters_add_forecast(y_train, y_test, seasonal)
    models.append(hw_add)
    hw_mult = holt_winters_mult_forecast(y_train, y_test, seasonal)
    models.append(hw_mult)
    arima = arima_forecast(y_train, y_test)
    models.append(arima)
    sarima = sarima_forecast(y_train, y_test, seasonal)
    models.append(sarima)
    return pd.DataFrame(models)


def get_features_and_target(df: pd.DataFrame, stock: str) -> tuple:
    X = df.drop(f'Close {stock}', axis=1)
    y = df[f'Close {stock}']
    return X, y


def get_features_and_target_splitted(df: pd.DataFrame, stock: str) -> tuple:
    X, y = get_features_and_target(df, stock)
    X_train, X_test = custom_train_test(X)
    y_train, y_test = custom_train_test(y)
    return X_train, X_test, y_train, y_test


def ml_pipeline(df: pd.DataFrame, stock: str):
    X_train, X_test, y_train, y_test = get_features_and_target_splitted(df, stock)
    tscv = TimeSeriesSplit(n_splits=15)  # Cross validation com time series
    modelos = {
        'DecisionTree': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 5]
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 5, 10],
                'ccp_alpha': [0.1, 0.01, 1]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 10],
                'learning_rate': [0.01, 0.1]
            }
        },
        'LightGBM': {
            'model': LGBMRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'num_leaves': [31, 50],
                'learning_rate': [0.01, 0.1]
            }
        }
    }
    best_models = {}

    for nome, config in modelos.items():
        print(f"Treinando {nome}...")
        grid = GridSearchCV(config['model'],
                            config['params'],
                            cv=tscv,
                            scoring='r2',
                            n_jobs=-1)
        grid.fit(X_train, y_train)
        forecast = grid.predict(X_test)
        mape, rmse, r2 = get_metrics(y_test, forecast)
        best_models[nome] = {
            'melhor_estimator': grid.best_estimator_,
            'melhor_score': grid.best_score_,
            'melhores_params': grid.best_params_,
            'modelo': grid,
            'predict': forecast,
            'mape_test': mape,
            'rmse_test': rmse,
            'r2': r2
        }
    
    return best_models