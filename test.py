import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from tqdm import tqdm

from application.main.database.entities.correlation import Correlation
from application.main.database.entities.stock_symbols import Company
from application.main.requests.indicators_request import load_indicator_ticks
from application.main.services.correlation_services import query_and_filter
from application.main.services.training_services import query_to_ts, train

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

    from application.main.requests.stock_request import load_stock_ticks

    # load_ticks(
    #     func="TIME_SERIES_INTRADAY",
    #     interval="15min",
    #     symbol="MSFT",
    #     start_month="2000-01",
    # )

    # df = pd.read_csv("sp500_companies.csv")
    # df = df.loc[df["Sector"].isin(["Technology", "Communication Services"])]
    # for interval in ["15min"]:
    #     for index, row in tqdm(df.iterrows(), total=df.shape[0], position=0, leave=True):
    #         load_ticks(
    #             func="TIME_SERIES_INTRADAY",
    #             interval=interval,
    #             symbol=row["Symbol"],
    #             start_month="2010-01",
    #         )
    #
    # load_indicator_ticks(
    #     func="BBANDS",
    #     symbol="QCOM",
    #     interval="15min",
    #     start_month="2010-01",
    #     time_period="10",
    #     series_type="close",
    #     nbdevup="1.5",
    #     nbdevdn="1.5",
    #     fastperiod="12",
    #     slowperiod="26",
    #     signalperiod="9"
    # )
    #
    # load_dotenv()
    #
    # engine = create_engine(os.environ["SQLITE_LOCATION"], echo=False)
    # Session = sessionmaker(bind=engine, expire_on_commit=True)
    #
    # # Define the base class
    # Base = declarative_base()
    #
    # class Correlation(Base):
    #     __tablename__ = 'correlations'
    #     id = Column(Integer, primary_key=True)
    #     symbol_1 = Column(String(10), nullable=False)
    #     symbol_2 = Column(String(10), nullable=False)
    #     correlation = Column(Float, nullable=False)
    #     p_value = Column(Float, nullable=False)
    #     target_field = Column(String(10))
    #
    # Base.metadata.create_all(engine)

    # stock_tick("TIME_SERIES_DAILY_30min_AAPL").__table__.drop(engine)
    #
    # df = query_and_filter("QCOM", function=None, interval=None, sort='abs_asc')
    # print(df)

    date_format = "%Y-%m-%d"
    train(
        func="TIME_SERIES_INTRADAY",
        interval="15min",
        target_symbol="QCOM",
        period=(
            datetime.strptime("2010-01-01", date_format),
            datetime.strptime("2022-01-01", date_format)
        ),
        # testing_period=(
        #     datetime.strptime("2022-01-01", date_format),
        #     datetime.now()
        # ),
        # testing_period=None,
        correlated_symbols=None,
        indicator_settings={
            "MACD": {"slowperiod": 26},
            "BBANDS": {"dev": 1.5},
        },
        time_period="20",
        series_type="close",
        nbdevup="1.5",
        nbdevdn="1.5",
        fastperiod="12",
        slowperiod="26",
        signalperiod="9",
        lstm_layers=[128, 64, 32], fc_layers=[16, 8], dropout=0.1
    )
