from typing import Optional, List

import pandas as pd
from keras.src.utils.module_utils import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing import timeseries_dataset_from_array

from application.main.utils.timeframe_processing import process_timeframe


def generate_trend_ts(
    df_target_asset: pd.DataFrame,
    df_correlated_asset: Optional[List[pd.DataFrame]] = None,
    df_indicators: Optional[List[pd.DataFrame]] = None,
    target_field: str = "close",
    sequence_length: int = 60,
    batch_size: int = 32,
    scaling_range: (int, int) = (-1, 1),
    shuffle: bool = False,
) -> Optional[tf.data.Dataset]:
    super().__init__()

    # Get the trend target (long/short)
    df_target_asset["trend"] = (
        df_target_asset[target_field] <= df_target_asset[target_field].shift(-1)
    ).astype(int)
    df_target_asset = df_target_asset[:-1]

    # Crop and match the timestamp
    x = [df_target_asset, *df_correlated_asset, *df_indicators]
    x = process_timeframe(x)

    # Gather the y (target)
    y = x[0]["trend"].to_numpy()

    # Drop the target and merge all
    x[0].drop(columns=["trend"], inplace=True)
    for df in x[:1]:
        x[0] = pd.merge(x[0], df, left_index=True, right_index=True)

    # Drop timestamp
    x[0].drop(columns=["timestamp"], inplace=True)
    x[0] = df_target_asset.reset_index(drop=True).to_numpy()

    scaler = MinMaxScaler(feature_range=scaling_range)

    # Gather the x
    x = scaler.fit_transform(x[0])

    return timeseries_dataset_from_array(
        x[sequence_length],
        y[sequence_length - 1 :],
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=shuffle,
    )
