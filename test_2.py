import pandas as pd

from application.main.services.correlation_services import generate_correlation

if __name__ == '__main__':
    df = pd.read_csv("sp500_companies.csv")
    df = df.loc[df["Sector"].isin(["Technology", "Communication Services"])]["Symbol"].to_list()

    generate_correlation(
        func="TIME_SERIES_INTRADAY",
        interval="15min",
        symbols=df,
        target_field="close"
    )
