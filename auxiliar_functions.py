import pandas as pd
import requests
import yfinance as yf

def get_ipca_data() -> pd.DataFrame:
    df = pd.read_excel('data/ipca-ibge.xlsx') 
    df.drop(columns=["IPCA", "Fator"], inplace=True)
    df = df[df["Data"] >= "2017-01-01"]
    daily_index = pd.date_range(start=df["Data"].min(), end=df["Data"].max(), freq="D")
    daily_df = pd.DataFrame({"Data": daily_index})
    df["AnoMes"] = df["Data"].dt.to_period("M")
    daily_df["AnoMes"] = daily_df["Data"].dt.to_period("M")
    merged = daily_df.merge(df[["AnoMes", "IPCA 12 meses"]], on="AnoMes", how="left")
    merged.drop(columns=["AnoMes"], inplace=True)
    merged.columns = ["Date", "IPCA"]
    return merged


def get_selic_data() -> pd.DataFrame:
    df = pd.read_csv('data\selic-bacen.csv', sep=';', encoding='latin1')
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
    df["Selic Anualizada"] = df["Selic Anualizada"].str.replace(',', '.').astype(float)
    df.drop("11 - Taxa de juros - Selic - % a.d.", axis=1, inplace=True)
    df.columns = ["Date", "SELIC"]
    return df


def get_dollar_data() -> pd.DataFrame:
    base_url = "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)?@dataInicial='01-01-2017'&@dataFinalCotacao='12-31-2024'&$top=100000&$format=json"
    response = requests.get(base_url).json()
    df = pd.DataFrame(response["value"])
    df.drop("cotacaoVenda", axis=1, inplace=True)
    df["dataHoraCotacao"] = pd.to_datetime(df["dataHoraCotacao"])
    df["dataHoraCotacao"] = pd.to_datetime(df["dataHoraCotacao"].dt.strftime("%Y-%m-%d"))
    df = df[["dataHoraCotacao", "cotacaoCompra"]]
    df.columns = ["Date", "Cambio"]
    return df

def get_stocks_data(stock_name: str, starting_year: int = 2017, ending_year: int = 2024) -> pd.DataFrame:
    """
    Get close price and volume data from stock.

    Args:
        stock_name (str): The stock symbol.
        starting_year (int): The year to start collecting data.
        ending_year (int): The year to stop collecting data.

    Return:
        pd.Series: close prices from stock.
    """
    stock_name = stock_name.upper()
    df = yf.Ticker(f'{stock_name}.SA').history(start=f'{starting_year}-01-01', end=f'{ending_year}-12-31')
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d"))
    df.rename(columns={
        "Close": f"Close {stock_name}",
        "Volume": f"Volume {stock_name}"
    }, inplace=True)
    return df[["Date", f"Close {stock_name}", f"Volume {stock_name}"]]

def concat_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, df4: pd.DataFrame) -> pd.DataFrame:
    return df1.merge(df2, on="Date", how="left").merge(df3, on="Date", how="left").merge(df4, on="Date", how="left")

def get_multiple_stocks_data(stock_list: list[str]) -> pd.DataFrame:
    stocks_dfs = [get_stocks_data(stock) for stock in stock_list]

    final_df = stocks_dfs[0]
    for df in stocks_dfs[1:]:
        final_df = final_df.merge(df, on="Date", how="left")

    return final_df
