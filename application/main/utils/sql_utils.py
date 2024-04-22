from typing import Optional
from .indicator_processors.indicator_processor import indicator_required_settings


def get_table_name(func: str, interval: Optional[str], symbol: str, **kwargs) -> str:
    # Check if the interval should result in a simple table name without the interval
    if interval in ["daily", "weekly", "monthly", None]:
        base_name = "_".join([func, symbol])
    else:
        base_name = "_".join([func, interval, symbol])

    # General handling for any function listed in indicators
    if func in indicator_required_settings:
        Model = indicator_required_settings[func]  # Retrieve the Pydantic model class
        try:
            # Create an instance of the model using kwargs
            model_instance = Model(**kwargs)
            # Extract model data to append to the base name
            model_data = model_instance.dict(exclude_none=True)  # Exclude None values
            # Append each available parameter to the base name
            for value in model_data.values():
                base_name += f"_{value}"
        except Exception as e:
            raise ValueError(f"Error processing data for {func}: {e}")

    return base_name
