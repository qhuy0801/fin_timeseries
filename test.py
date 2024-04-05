import pandas as pd
from tqdm import tqdm

from application.main.database.entities.stock_symbols import Company

if __name__ == "__main__":
    # from application.main.database import SessionLocal
    #
    # with SessionLocal() as session:
    #     df = pd.read_csv("sp500_companies.csv")
    #     for index, row in df.iterrows():
    #         company = Company(
    #             exchange=row["Exchange"],
    #             symbol=row["Symbol"],
    #             shortname=row["Shortname"],
    #             longname=row["Longname"],
    #             sector=row["Sector"],
    #             industry=row["Industry"],
    #             city=row["City"],
    #             state=row["State"],
    #             country=row["Country"],
    #             business_summary=row["Longbusinesssummary"],
    #             weight=row["Weight"],
    #         )
    #         session.add(company)
    #     session.commit()
    #     session.close()

    from application.main.requests.alpha_vantage import load_ticks

    # load_ticks(
    #     func="TIME_SERIES_INTRADAY",
    #     interval="15min",
    #     symbol="MSFT",
    #     start_month="2000-01",
    # )

    df = pd.read_csv("sp500_companies.csv")
    df = df.loc[df["Sector"].isin(["Technology", "Communication Services"])]
    for interval in ["5min", "15min", "30min"]:
        for index, row in tqdm(df.iterrows(), total=df.shape[0], position=0, leave=True):
            load_ticks(
                func="TIME_SERIES_INTRADAY",
                interval=interval,
                symbol=row["Symbol"],
                start_month="2010-01",
            )
