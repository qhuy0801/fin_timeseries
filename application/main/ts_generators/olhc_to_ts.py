from typing import Optional, List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing import timeseries_dataset_from_array


class OLHC_Trend_TSGenerator:
    def __init__(
        self,
        df_target_asset: pd.DataFrame,
        df_correlated_asset: Optional[List[pd.DataFrame]] = None,
        df_indicators: Optional[List[pd.DataFrame]] = None,
        target_field: str = "close",
        sequence_length: int = 60,
        batch_size: int = 32,
        scaling_range: (int, int) = (-1, 1),
    ) -> None:
        super().__init__()

        # Initialize with the timeframe of df_target_asset
        start_time = df_target_asset["timestamp"].min()
        end_time = df_target_asset["timestamp"].max()

        df_target_asset["trend"] = (
            df_target_asset[target_field] <= df_target_asset[target_field].shift(-1)
        ).astype(int)

        df_target_asset = df_target_asset[:-1]

        # Update the timeframe based on df_correlated_asset and df_indicators
        if df_correlated_asset is not None:
            for index, df in enumerate(df_correlated_asset):
                df = df.rename(
                    columns={
                        col: f"{col}_{index}"
                        for col in df.columns
                        if col != "timestamp"
                    }
                )
                start_time = max(start_time, df["timestamp"].min())
                end_time = min(end_time, df["timestamp"].max())

        if df_indicators is not None:
            for index, df in enumerate(df_indicators):
                df = df.rename(
                    columns={
                        col: f"{col}_{index}"
                        for col in df.columns
                        if col != "timestamp"
                    }
                )
                start_time = max(start_time, df["timestamp"].min())
                end_time = min(end_time, df["timestamp"].max())

        # Crop df_target_asset
        df_target_asset = df_target_asset[
            (df_target_asset["timestamp"] >= start_time)
            & (df_target_asset["timestamp"] <= end_time)
        ]

        # Crop each DataFrame in df_correlated_asset and df_indicators
        if df_correlated_asset is not None:
            df_correlated_asset = [
                df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]
                for df in df_correlated_asset
            ]

        if df_indicators is not None:
            df_indicators = [
                df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]
                for df in df_indicators
            ]

        # Gather the y
        self.y = df_target_asset["trend"].to_numpy()

        # Concat all dataframes and scale
        # Concatenate correlated assets, if any
        if df_correlated_asset is not None:
            for df in df_correlated_asset:
                df_target_asset = pd.merge(
                    df_target_asset, df, on="timestamp", how="left"
                )

        # Concatenate indicators, if any
        if df_indicators is not None:
            for df in df_indicators:
                df_target_asset = pd.merge(
                    df_target_asset, df, on="timestamp", how="left"
                )

        df_target_asset.drop(columns=["timestamp", "trend"], inplace=True)
        df_target_asset = df_target_asset.to_numpy()

        scaler = MinMaxScaler(feature_range=scaling_range)

        # Gather the x
        self.x = scaler.fit_transform(df_target_asset)

        # Other setting
        self.sequence_length = sequence_length
        self.batch_size = batch_size

    def get_ts_generator(self, shuffle: bool = False):
        return timeseries_dataset_from_array(
            self.x[:-self.sequence_length],
            self.y[self.sequence_length-1:],
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

