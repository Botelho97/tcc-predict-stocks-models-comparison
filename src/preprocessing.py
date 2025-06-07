import pandas as pd


def read_csv(path: str ="data/stocks_info.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df


def filter_df(stock: str, df: pd.DataFrame) -> pd.DataFrame:
    return df[[f'Close {stock}', f'Volume {stock}', 'SELIC', 'IPCA', 'Cambio']]


def feature_engineering(stock: str, df: pd.DataFrame) -> pd.DataFrame:
    df["Lag 1"] = df[f"Close {stock}"].shift(1)
    df["Lag 3"] = df[f"Close {stock}"].shift(3)
    df["Lag 5"] = df[f"Close {stock}"].shift(5)
    df["ma_5d"] = df[f"Close {stock}"].rolling(window=5).mean()
    df["ma_10d"] = df[f"Close {stock}"].rolling(window=10).mean()
    df["ma_20d"] = df[f"Close {stock}"].rolling(window=20).mean()
    return df


def final_df(stock: str) -> pd.DataFrame:
    stock = stock.upper()
    df = read_csv()
    df = filter_df(stock, df)
    df = feature_engineering(stock, df)
    return df
