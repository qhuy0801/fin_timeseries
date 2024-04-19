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
    correlated_symbols: Optional[List[str]],
    indicator_funcs: Optional[List[str]],
    **kwargs,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    List[pd.DataFrame] | None,
    List[pd.DataFrame] | None,
    List[pd.DataFrame] | None,
    List[pd.DataFrame] | None,
]:
    """
    Query the target symbol data, correlated data and technical indicators
    Args:
        func (str): Timeframe TIME_SERIES_INTRADAY/ TIME_SERIES_DAILY
        interval (str): 1min, 5min, 15min, 30min, 60min
        target_symbol (str): The asset symbol
        training_period (Optional[Tuple[datetime, datetime]]): If None provided then the default is from 2010-01-01 to now
        testing_period (Optional[Tuple[datetime, datetime]]): If None provided then there the testing output will be None
        correlated_symbols (Optional[List[str]]): The assets that have high correlation to the target asset
        indicator_funcs (Optional[List[str]]): List of indicator which serve the training purpose
        **kwargs: Other supported parameters for the indicators

    Returns:
        target_training (pandas.DataFrame)
        target_testing (Optional[List[pandas.DataFrame]])
        correlated_training (Optional[List[pandas.DataFrame]])
        correlated_testing (Optional[List[pandas.DataFrame]])
        indicator_training (Optional[List[pandas.DataFrame]])
        indicator_testing (Optional[List[pandas.DataFrame]])
    """
    # Process the timeframe
    if not training_period:
        training_period = (datetime(2010, 1, 1), datetime.now())

    # Gather the target data
    _target_symbol = stock_tick(Base, get_table_name(func, interval, target_symbol))

    # Target symbol
    target_training = None
    target_testing = None
    with Session() as session:
        target_training = pd.read_sql(
            session.query(_target_symbol)
            .filter(
                _target_symbol.timestamp >= training_period[0],
                _target_symbol.timestamp <= training_period[1],
            )
            .order_by(_target_symbol.timestamp)
            .statement,
            session.bind,
        )
        if testing_period:
            target_testing = pd.read_sql(
                session.query(_target_symbol)
                .filter(
                    _target_symbol.timestamp >= testing_period[0],
                    _target_symbol.timestamp <= testing_period[1],
                )
                .order_by(_target_symbol.timestamp)
                .statement,
                session.bind,
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
                .order_by(correlated_symbol.timestamp)
                .statement,
                session.bind,
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
                    .order_by(correlated_symbol.timestamp)
                    .statement,
                    session.bind,
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
            getattr(indicators, indicator_func.split("_")[0].lower() + f"_tick", None)(
                Base, indicator_func
            )
            for indicator_func in indicator_funcs
        ]
        indicator_training = [
            pd.read_sql(
                session.query(indicator_func)
                .filter(
                    indicator_func.timestamp >= training_period[0],
                    indicator_func.timestamp <= training_period[1],
                )
                .order_by(indicator_func.timestamp)
                .statement,
                session.bind,
            )
            for indicator_func in indicator_funcs
        ]
        if testing_period:
            indicator_testing = [
                pd.read_sql(
                    session.query(indicator_func)
                    .filter(
                        indicator_func.timestamp >= testing_period[0],
                        indicator_func.timestamp <= testing_period[1],
                    )
                    .order_by(indicator_func.timestamp)
                    .statement,
                    session.bind,
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
