from typing import Optional

_indicators = {
    "SMA": ["time_period", "series_type"],
    "EMA": ["time_period", "series_type"],
    "MACD": ["series_type", "fastperiod", "slowperiod", "signalperiod"],
    "BBANDS": ["time_period", "series_type", "nbdevup", "nbdevdn"]
}


def get_table_name(func: str, interval: Optional[str], symbol: str, **kwargs) -> str:
    # Check if the interval should result in a simple table name without the interval
    if interval in ["daily", "weekly", "monthly", None]:
        base_name = "_".join([func, symbol])
    else:
        base_name = "_".join([func, interval, symbol])

    # General handling for any function listed in indicators
    if func in _indicators:
        params = _indicators[func]
        # Append each parameter to the base name, fetching from kwargs or using 'default'
        for param in params:
            value = kwargs.get(param)
            base_name += f"_{value}"

    return base_name
