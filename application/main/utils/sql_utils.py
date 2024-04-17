from typing import Optional


def get_table_name(func: str, interval: Optional[str], symbol: str, **kwargs) -> str:
    if interval in ["daily, weekly, monthly", None]:
        return "_".join([func, symbol])
    else:
        return "_".join([func, interval, symbol])
