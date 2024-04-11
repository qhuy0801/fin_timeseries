"""
The module for processing the trading timeframe
"""
import pandas as pd


def timeframe_cropping(df1: pd.DataFrame, df2: pd.DataFrame, index_column: str = "timestamp") -> (pd.DataFrame, pd.DataFrame):
    """
    Crop two DataFrames to the common timeframe that both DataFrames cover,
    with indices in "YYYY-MM-DD" format.

    Parameters:
    - df1: First DataFrame with index in "YYYY-MM-DD" format.
    - df2: Second DataFrame with index in "YYYY-MM-DD" format.

    Returns:
    - Tuple of DataFrames cropped to the common timeframe.
    """
    # Ensure the index is in datetime format
    df1, df2 = df1.set_index(index_column), df2.set_index(index_column)
    df1.index = pd.to_datetime(df1.index)
    df2.index = pd.to_datetime(df2.index)

    # Find the common start and end dates
    common_start = max(df1.index.min(), df2.index.min())
    common_end = min(df1.index.max(), df2.index.max())

    # Crop both DataFrames to the common timeframe
    df1_cropped = df1[(df1.index >= common_start) & (df1.index <= common_end)]
    df2_cropped = df2[(df2.index >= common_start) & (df2.index <= common_end)]

    # Align indices to ensure they match by using intersection of indices
    common_indices = df1_cropped.index.intersection(df2_cropped.index)
    df1_aligned = df1_cropped.reindex(common_indices)
    df2_aligned = df2_cropped.reindex(common_indices)

    return df1_aligned, df2_aligned
