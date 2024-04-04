import pandas as pd

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

    df = pd.read_csv("sp500_companies.csv")
    for interval in ["1min", "5min", "15min", "30min"]:
        for index, row in df.iterrows():
            load_ticks(
                func="TIME_SERIES_INTRADAY",
                interval=interval,
                symbol=row["Symbol"],
                start_month="2000-01",
            )
