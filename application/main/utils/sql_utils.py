from typing import Optional

_indicators = {
    "sma": ["time_period", "series_type"],
    "ema": ["time_period", "series_type"],
    "macd": ["series_type", "fastperiod", "slowperiod", "signalperiod"],
    "bbands": ["time_period", "series_type", "nbdevup", "nbdevdn"]
}


def get_table_name(func: str, interval: Optional[str], symbol: str, **kwargs) -> str:
    # Check if the interval should result in a simple table name without the interval
    if interval in ["daily", "weekly", "monthly", None]:
        base_name = "_".join([func, symbol])
    else:
        base_name = "_".join([func, interval, symbol])

    # General handling for any function listed in indicators
    if func.lower() in _indicators:
        params = _indicators[func]
        # Append each parameter to the base name, fetching from kwargs or using 'default'
        for param in params:
            value = kwargs.get(param)
            base_name += f"_{value}"

    return base_name
