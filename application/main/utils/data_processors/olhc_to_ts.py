from typing import Optional, List, Any, Union, Generator, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from timeframe_processing import process_timeframe


def trend_ts(
    df_target_asset: pd.DataFrame,
    df_correlated_asset: Optional[List[pd.DataFrame]] = None,
    df_indicators: Optional[List[pd.DataFrame]] = None,
    target_field: str = "close",
    sequence_length: int = 60,
    to_generator: bool = True,
    **kwargs: Any,
) -> Union[Generator[Tuple[np.ndarray, np.ndarray], None, None], Tuple[np.ndarray, np.ndarray]]:
    """
    Combine all dataframe into one time series
    Args:
        df_target_asset:
        df_correlated_asset:
        df_indicators:
        target_field:
        sequence_length:
        to_generator (bool): If true, the output will be an generator, else, it will return 2 numpy array in the format
            (data, label)
        **kwargs:

    Returns:
        Union[Generator[Tuple[np.ndarray, np.ndarray], None, None], Tuple[np.ndarray, np.ndarray]]:
        Either an generator that generate tuple(target, label) or 2 numpy array in format (data, label)

    """
    # Get the trend target (long/short)
    df_target_asset["trend"] = (
        df_target_asset[target_field] <= df_target_asset[target_field].shift(-1)
    ).astype(int)
    df_target_asset = df_target_asset[:-1]

    # Crop and match the timestamp
    x = [df_target_asset] + [
        item
        for sublist in (df_correlated_asset, df_indicators)
        if sublist is not None
        for item in sublist
    ]
    x = process_timeframe(x)

    # Gather the y (target)
    y = x[0]["trend"].to_numpy()

    # Drop the target and merge all
    x[0].drop(columns=["trend"], inplace=True)
    for df in x[1:]:
        x[0] = pd.merge(x[0], df, left_index=True, right_index=True)

    # Convert to numpy
    x = x[0].reset_index(drop=True)

    # Gather the x
    x[x.columns] = StandardScaler().fit_transform(x[x.columns])
    x = x.to_numpy()

    if to_generator:
        # Generator expression
        return (
            (x[i : i + sequence_length], y[i + sequence_length - 1])
            for i in range(len(x) - sequence_length)
        )
    else:
        # Process the data into arrays
        X, Y = [], []
        for i in range(len(x) - sequence_length):
            X.append(x[i : i + sequence_length])
            Y.append(y[i + sequence_length - 1])
        return np.array(X), np.array(Y)