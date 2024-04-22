from typing import List, Dict

import pandas as pd
from pydantic import ValidationError

from ta.trend import SMAIndicator as SMA, EMAIndicator as EMA, MACD
from ta.volatility import BollingerBands as BBANDS

from application.main.utils.indicator_processors.indicator_setting import (
    indicator_required_settings,
)


def generate_indicator(
    data: pd.DataFrame, indicator_settings: Dict[str, Dict[str, int | str]]
) -> List[pd.DataFrame]:
    indicator_values = pd.DataFrame()

    for indicator_name in indicator_settings:
        model_class = indicator_required_settings.get(indicator_name)
        if not model_class:
            raise ValueError(f"No settings found for indicator '{indicator_name}'")

        # Create model instance with provided kwargs, filling with defaults
        try:
            indicator = model_class(**indicator_settings.get(indicator_name))
        except ValidationError as e:
            raise ValueError(
                f"Validation error for {indicator_name} settings: {str(e)}"
            )

        series = data.get(
            indicator.series_type, data["close"]
        )

        # Calculate the indicator using settings
        try:
            if indicator_name == "SMA":
                indicator_values["sma"] = SMA(
                    series, window=indicator.time_period
                ).sma_indicator()
            elif indicator_name == "EMA":
                indicator_values["ema"] = EMA(
                    series, window=indicator.time_period
                ).ema_indicator()
            elif indicator_name == "MACD":
                macd = MACD(
                    series,
                    window_fast=indicator.fastperiod,
                    window_slow=indicator.slowperiod,
                    window_sign=indicator.signalperiod,
                )
                indicator_values["macd"] = macd.macd()
                indicator_values["macd_hist"] = macd.macd_diff()
                indicator_values["macd_signal"]= macd.macd_signal()
            elif indicator_name == "BBANDS":
                bbands = BBANDS(
                    series,
                    window=indicator.time_period,
                    window_dev=indicator.dev,
                )
                indicator_values['upperband'] = bbands.bollinger_hband()
                indicator_values['middleband'] = bbands.bollinger_mavg()
                indicator_values['lowerband'] = bbands.bollinger_lband()
            else:
                raise ValueError(f"Unsupported indicator '{indicator_name}'")
        except ValueError as e:
            print("Unsupported indicator skipped!")
            continue

    return indicator_values
