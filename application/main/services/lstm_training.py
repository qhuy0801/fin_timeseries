import importlib
import os
from datetime import datetime

from typing import Optional, List, Tuple, Generator, Dict

import joblib
import pandas as pd
import wandb
from dotenv import load_dotenv
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from tensorflow.python.keras.saving.save import load_model
from wandb.integration.keras import WandbMetricsLogger
from wandb.sdk.wandb_run import Run

from application.main.database.entities.stock_tick import stock_tick
from application.main.utils import generate_indicator
from application.main.utils import get_table_name
from application.main.utils.data_processors.class_balancer import auto_resample

load_dotenv()

engine = create_engine(os.environ["SQLITE_LOCATION"], echo=False)
Session = sessionmaker(bind=engine, expire_on_commit=True)

Base = declarative_base()

indicators_path = "application.main.database.entities.indicators"
indicators = importlib.import_module(indicators_path)

models_path = "application.main.models"
models = importlib.import_module(models_path)

data_processor_path = "application.main.utils"
data_processors = importlib.import_module(data_processor_path)


def query_to_ts(
    func: str,
    interval: str,
    target_symbol: str,
    period: Optional[Tuple[datetime, datetime]],
    correlated_symbols: Optional[List[str]] = None,
    indicator_settings: Optional[Dict[str, Dict[str, int | str]]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, List[pd.DataFrame] | None]:
    """
    Query the target symbol data, correlated data and technical indicators
    Args:
        func (str): Timeframe TIME_SERIES_INTRADAY/ TIME_SERIES_DAILY
        interval (str): 1min, 5min, 15min, 30min, 60min
        target_symbol (str): The asset symbol
        period (Optional[Tuple[datetime, datetime]]): If None provided then the default is from 2010-01-01 to now
        correlated_symbols (Optional[List[str]]): The assets that have high correlation to the target asset
        indicator_settings (Optional[Dict[str, Dict[str, int | str]]]): Specify the indicators
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

    # Technical indicators
    indicator_data = None
    if indicator_settings is not None:
        indicator_data = generate_indicator(
            data=target_data, indicator_settings=indicator_settings
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

    return (
        target_data,
        indicator_data,
        correlated_data,
    )


def train(
    func: str,
    interval: str,
    target_symbol: str,
    period: Optional[Tuple[datetime, datetime]] = None,
    correlated_symbols: Optional[List[str]] = None,
    indicator_settings: Optional[Dict[str, Dict[str, int | str]]] = None,
    to_generator: bool = False,
    batch_size: int = 200,
    upsampling: Optional[bool] = None,
    model_name: str = "trend_lstm",
    sequence_length: int = 60,
    validation_size: Optional[float] = None,
    shuffle: bool = True,
    epochs: int = 10,
    wandb_log: Optional[Run] = None,
    model_registry_name: Optional[str] = None,
    **kwargs,
):
    # Query the data
    (
        target_data,
        indicator_data,
        correlated_data,
    ) = query_to_ts(
        func=func,
        interval=interval,
        target_symbol=target_symbol,
        period=period,
        correlated_symbols=correlated_symbols,
        indicator_settings=indicator_settings,
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
        _, x, y, scaler = data
        feature_count = x[0].shape[-1]

        # Validation
        if validation_size is not None:
            x, x_val, y, y_val = train_test_split(
                x, y, test_size=validation_size, random_state=42
            )

        # Up-sampling if needed
        if upsampling is not None:
            x, y = auto_resample(x=x, y=y, upsample=upsampling)

        # Model object
        model = getattr(models, model_name, None)
        if not model:
            raise ValueError(f"Model {model_name} not found!.")
        model = model(
            sequence_length=sequence_length, feature_count=feature_count, **kwargs
        )

        # Training settings
        callbacks = [
            ReduceLROnPlateau(
                monitor="val_loss" if validation_size is not None else "loss",
                factor=0.95,
                patience=30,
                mode="max",
                min_delta=0.0005,
            ),
            EarlyStopping(
                monitor="val_loss" if validation_size is not None else "loss",
                mode="max",
                min_delta=0.0001,
                patience=100,
            ),
        ]

        if wandb_log:
            callbacks = callbacks + [
                WandbMetricsLogger(),
            ]

        model.fit(
            x=x,
            y=y,
            shuffle=shuffle,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val) if validation_size is not None else None,
            callbacks=callbacks,
            verbose=1,
            validation_freq=1,
        )

        # Saving and logging model to cloud
        _model_path = f"model.keras"
        model.save(_model_path)

        if wandb_log is not None:
            artifact = wandb.Artifact(
                name=f"{wandb_log.name}_ckpt", type="model_n_scaler"
            )

            # Add model
            artifact.add_file(local_path=_model_path)

            # Add scaler
            if isinstance(scaler, StandardScaler):
                scaler_path = f"scaler.save"
                joblib.dump(value=scaler, filename=scaler_path)
                artifact.add_file(local_path=scaler_path)

            wandb_log.log_artifact(artifact)

            # Link model to registry
            if model_registry_name is not None:
                wandb_log.link_model(
                    path=_model_path, registered_model_name=model_registry_name
                )


def inferent(
    func: str,
    interval: str,
    target_symbol: str,
    model_path: str,
    model_name: str,
    scaler_path: str,
    period: Optional[Tuple[datetime, datetime]] = None,
    correlated_symbols: Optional[List[str]] = None,
    indicator_settings: Optional[Dict[str, Dict[str, int | str]]] = None,
    sequence_length: int = 60,
    **kwargs,
):
    # Load the model
    model = load_model(model_path, compile=False)

    # Query the data
    (
        target_data,
        indicator_data,
        correlated_data,
    ) = query_to_ts(
        func=func,
        interval=interval,
        target_symbol=target_symbol,
        period=period,
        correlated_symbols=correlated_symbols,
        indicator_settings=indicator_settings,
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
        scaler_path=scaler_path,
        **kwargs,
    )

    # Unpact the data
    timestamp, x, y, _ = data

    # Get empty dataframe to store the prediction
    result_table = []

    for _timestamp, _x, _y in zip(timestamp, x, y):
        [_y_pred] = model([x], training=False)
        _data = {
            "timestamp": _timestamp,
            "predicted_prob": _y_pred,
            "ground_truth": _y,
        }
        result_table.append(_data)
    result_table = pd.DataFrame(result_table)

    return result_table
