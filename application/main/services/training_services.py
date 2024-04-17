import os
from datetime import datetime

from typing import Optional, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from application.main.database.entities.stock_tick import stock_tick
from application.main.utils.sql_utils import get_table_name

load_dotenv()

engine = create_engine(os.environ["SQLITE_LOCATION"], echo=False)
Session = sessionmaker(bind=engine, expire_on_commit=True)

Base = declarative_base()


def query_to_ts(
    func: str,
    interval: str,
    target_symbol: str,
    training_period: Tuple[datetime, datetime],
    testing_period: Optional[Tuple[datetime, datetime]],
    validation_ratio: Optional[float],
    correlated_symbols: Optional[List[str]],
    indicators=Optional[List[str]],
):
    # Gather the data
    target_table = get_table_name(func, interval, target_symbol)

    # Get the table_names TODO: indicators
    correlated_data = None
    if correlated_symbols:
        correlated_data = [
            get_table_name(func, interval, symbol) for symbol in correlated_symbols
        ]

    # Query
    with Session() as session:
        target_table = stock_tick(Base, target_table)
        target_training = pd.read_sql(
            session.query(target_table)
            .filter(
                target_table.timestamp >= training_period[0],
                target_table.timestamp <= training_period[1],
            )
            .order_by(target_table.timestamp),
            session.bind(),
        )

        if testing_period:
            target_testing = pd.read_sql(
                session.query(target_table)
                .filter(
                    target_table.timestamp >= testing_period[0],
                    target_table.timestamp <= testing_period[1],
                )
                .order_by(target_table.timestamp),
                session.bind(),
            )

    return None
