import importlib
import os
from datetime import datetime

from typing import Optional, List, Tuple, Generator

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from wandb.sdk.wandb_run import Run

from application.main.database.entities.stock_tick import stock_tick
from application.main.utils.sql_utils import get_table_name

load_dotenv()

engine = create_engine(os.environ["SQLITE_LOCATION"], echo=False)
Session = sessionmaker(bind=engine, expire_on_commit=True)

Base = declarative_base()

indicators_path = "application.main.database.entities.indicators"
indicators = importlib.import_module(indicators_path)

model_path = "application.main.models"
models = importlib.import_module(model_path)

data_processor_path = "application.main.data_processors"
data_processors = importlib.import_module(data_processor_path)


def query_to_ts(
    func: str,
    interval: str,
    target_symbol: str,
    period: Optional[Tuple[datetime, datetime]],
    correlated_symbols: Optional[List[str]] = None,
    indicator_funcs: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, List[pd.DataFrame] | None, List[pd.DataFrame] | None]:
    """
    Query the target symbol data, correlated data and technical indicators
    Args:
        func (str): Timeframe TIME_SERIES_INTRADAY/ TIME_SERIES_DAILY
        interval (str): 1min, 5min, 15min, 30min, 60min
        target_symbol (str): The asset symbol
        period (Optional[Tuple[datetime, datetime]]): If None provided then the default is from 2010-01-01 to now
        correlated_symbols (Optional[List[str]]): The assets that have high correlation to the target asset
        indicator_funcs (Optional[List[str]]): List of indicator which serve the training purpose
        **kwargs: Other supported parameters for the indicators

    Returns:
        target_data (pandas.DataFrame)
        correlated_data (Optional[List[pandas.DataFrame]])
        indicator_data (Optional[List[pandas.DataFrame]])
    """
    # Process the timeframe
    if not period:
        period = (datetime(2010, 1, 1), datetime.now())

    # Gather the target data
    _target_symbol = stock_tick(Base, get_table_name(func, interval, target_symbol))

    # Target symbol
    target_data = None
    with Session() as session:
        target_data = pd.read_sql(
            session.query(_target_symbol)
            .filter(
                _target_symbol.timestamp >= period[0],
                _target_symbol.timestamp <= period[1],
            )
            .order_by(_target_symbol.timestamp)
            .statement,
            session.bind,
        )

    # Correlated symbols
    correlated_data = None
    if correlated_symbols:
        correlated_symbols = [
            get_table_name(func, interval, symbol) for symbol in correlated_symbols
        ]
        correlated_symbols = [
            stock_tick(Base, correlated_symbol)
            for correlated_symbol in correlated_symbols
        ]
        correlated_data = [
            pd.read_sql(
                session.query(correlated_symbol)
                .filter(
                    correlated_symbol.timestamp >= period[0],
                    correlated_symbol.timestamp <= period[1],
                )
                .order_by(correlated_symbol.timestamp)
                .statement,
                session.bind,
            )
            for correlated_symbol in correlated_symbols
        ]

    # Technical indicators
    indicator_data = None
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
        indicator_data = [
            pd.read_sql(
                session.query(indicator_func)
                .filter(
                    indicator_func.timestamp >= period[0],
                    indicator_func.timestamp <= period[1],
                )
                .order_by(indicator_func.timestamp)
                .statement,
                session.bind,
            )
            for indicator_func in indicator_funcs
        ]

    return (
        target_data,
        correlated_data,
        indicator_data,
    )


def train(
    func: str,
    interval: str,
    target_symbol: str,
    period: Optional[Tuple[datetime, datetime]] = None,
    correlated_symbols: Optional[List[str]] = None,
    indicator_funcs: Optional[List[str]] = None,
    to_generator: bool = False,
    model_name: str = "trend_lstm",
    sequence_length: int = 60,
    validation_size: Optional[float] = None,
    shuffle: bool = True,
    epochs: int = 10,
    wandb_log: Optional[Run] = None,
    **kwargs,
):
    # Query the data
    (
        target_data,
        correlated_data,
        indicator_data,
    ) = query_to_ts(
        func=func,
        interval=interval,
        target_symbol=target_symbol,
        period=period,
        correlated_symbols=correlated_symbols,
        indicator_funcs=indicator_funcs,
        **kwargs,
    )

    # Get the data processor
    data_processor = getattr(data_processors, f"{model_name.split('_')[0]}_ts", None)
    if not data_processor:
        raise ValueError(f"Processor {model_name.split('_')[0]}_ts not found!.")

    # Use the processor to process the data
    data = data_processor(
        df_target_asset=target_data,
        df_correlated_asset=correlated_data,
        df_indicators=indicator_data,
        sequence_length=sequence_length,
        to_generator=to_generator,
        **kwargs,
    )

    if isinstance(data, Generator):
        # Training with generator
        feature_count = next(data)[0].shape[1]

        # Model object
        model = getattr(models, model_name, None)
        if not model:
            raise ValueError(f"Model {model_name} not found!.")
        model = model(
            sequence_length=sequence_length, feature_count=feature_count, **kwargs
        )
        # TODO: start the training with generator
    else:
        # Training with numpy arrays
        x, y = data
        feature_count = x[0].shape[-1]

        # Model object
        model = getattr(models, model_name, None)
        if not model:
            raise ValueError(f"Model {model_name} not found!.")
        model = model(
            sequence_length=sequence_length, feature_count=feature_count, **kwargs
        )

        model.fit(
            x=x,
            y=y,
            shuffle=shuffle,
            epochs=epochs,
            validation_split=validation_size,
            callbacks=[
                WandbMetricsLogger(),
                WandbModelCheckpoint("models"),
            ] if wandb_log else None,
        )

        print(model)

    return None
