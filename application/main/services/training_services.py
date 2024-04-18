import importlib
import os
from datetime import datetime

from typing import Optional, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from application.main.database.entities.stock_tick import stock_tick
from application.main.utils.sql_utils import get_table_name

load_dotenv()

engine = create_engine(os.environ["SQLITE_LOCATION"], echo=False)
Session = sessionmaker(bind=engine, expire_on_commit=True)

Base = declarative_base()

indicators_path = "application.main.database.entities.indicators"
indicators = importlib.import_module(indicators_path)


def query_to_ts(
    func: str,
    interval: str,
    target_symbol: str,
    training_period: Optional[Tuple[datetime, datetime]],
    testing_period: Optional[Tuple[datetime, datetime]],
    # validation_ratio: Optional[float],
    correlated_symbols: Optional[List[str]],
    indicator_funcs=Optional[List[str]],
    **kwargs,
):
    """

    Args:
        func:
        interval:
        target_symbol:
        training_period:
        testing_period:
        correlated_symbols:
        indicator_funcs:
        **kwargs:

    Returns:

    """
    # Process the timeframe
    if not training_period:
        training_period = (datetime(2010, 1, 1), datetime.now())

    # Gather the target data
    target_symbol = get_table_name(func, interval, target_symbol)
    target_symbol = stock_tick(Base, target_symbol)

    # Target symbol
    target_training = None
    target_testing = None
    with Session() as session:
        target_training = pd.read_sql(
            session.query(target_symbol)
            .filter(
                target_symbol.timestamp >= training_period[0],
                target_symbol.timestamp <= training_period[1],
            )
            .order_by(target_symbol.timestamp),
            session.bind(),
        )
        if testing_period:
            target_testing = pd.read_sql(
                session.query(target_symbol)
                .filter(
                    target_symbol.timestamp >= testing_period[0],
                    target_symbol.timestamp <= testing_period[1],
                )
                .order_by(target_symbol.timestamp),
                session.bind(),
            )

    # Correlated symbols
    correlated_training = None
    correlated_testing = None
    if correlated_symbols:
        correlated_symbols = [
            get_table_name(func, interval, symbol) for symbol in correlated_symbols
        ]
        correlated_symbols = [
            stock_tick(Base, correlated_symbol)
            for correlated_symbol in correlated_symbols
        ]
        correlated_training = [
            pd.read_sql(
                session.query(correlated_symbol)
                .filter(
                    correlated_symbol.timestamp >= training_period[0],
                    correlated_symbol.timestamp <= training_period[1],
                )
                .order_by(correlated_symbol.timestamp),
                session.bind(),
            )
            for correlated_symbol in correlated_symbols
        ]
        if testing_period:
            correlated_testing = [
                pd.read_sql(
                    session.query(correlated_symbol)
                    .filter(
                        correlated_symbol.timestamp >= testing_period[0],
                        correlated_symbol.timestamp <= testing_period[1],
                    )
                    .order_by(correlated_symbol.timestamp),
                    session.bind(),
                )
                for correlated_symbol in correlated_symbols
            ]

    # Technical indicators
    indicator_training = None
    indicator_testing = None
    if indicator_funcs:
        indicator_funcs = [
            get_table_name(
                func=indicator_func,
                interval=interval,
                symbol=target_symbol,
                **kwargs,
            )
            for indicator_func in indicator_funcs
        ]
        indicator_funcs = [
            getattr(indicators, indicator_func.split("_")[0].lower() + f"_tick", None)
            for indicator_func in indicator_funcs
        ]
        indicator_training = [
            pd.read_sql(
                session.query(indicator_func)
                .filter(
                    indicator_func.time >= training_period[0],
                    indicator_func.time <= training_period[1],
                )
                .order_by(indicator_func.time),
                session.bind(),
            )
            for indicator_func in indicator_funcs
        ]
        if testing_period:
            indicator_testing = [
                pd.read_sql(
                    session.query(indicator_func)
                    .filter(
                        indicator_func.time >= testing_period[0],
                        indicator_func.time <= testing_period[1],
                    )
                    .order_by(indicator_func.time),
                    session.bind(),
                )
                for indicator_func in indicator_funcs
            ]

    return (
        target_training,
        target_testing,
        correlated_training,
        correlated_testing,
        indicator_training,
        indicator_testing,
    )
