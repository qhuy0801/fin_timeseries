import os

import sqlalchemy
from dotenv import load_dotenv
from application.main.database.entities.stock_tick import stock_tick
from application.main.database.factory.sqlite import engine

# load_dotenv()

if __name__ == '__main__':
    stock_tick("TIME_SERIES_DAILY_30min_AAPL").__table__.drop(engine)