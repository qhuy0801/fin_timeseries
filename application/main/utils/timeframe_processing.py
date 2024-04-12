"""
The module for processing the trading timeframe
"""
from typing import List

import pandas as pd


def process_timeframe(
    dataframes: List[pd.DataFrame], index_column: str = "timestamp"
) -> List[pd.DataFrame]:
    """

    Args:
        dataframes (List[pd.DataFrame]): List of DataFrames with index in "YYYY-MM-DD" format.
        index_column (str): Name of the column to use as the index, default 'timestamp'.

    Returns:
        List[pd.DataFrame]: List of DataFrames cropped to the common timeframe with matched indices.
    """

    # Set indices to the specified column and convert to datetime
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].set_index(index_column)
        dataframes[i].index = pd.to_datetime(dataframes[i].index)

    # Determine the common timeframe across all dataframes
    common_start = max(df.index.min() for df in dataframes)
    common_end = min(df.index.max() for df in dataframes)

    # Crop each dataframe to the common timeframe
    cropped_dataframes = [
        df[(df.index >= common_start) & (df.index <= common_end)] for df in dataframes
    ]

    # Find the intersection of indices from all dataframes
    common_indices = cropped_dataframes[0].index
    for df in cropped_dataframes[1:]:
        common_indices = common_indices.intersection(df.index)

    # Reindex all dataframes to the common indices
    aligned_dataframes = [df.reindex(common_indices) for df in cropped_dataframes]

    return aligned_dataframes
