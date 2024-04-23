from pydantic import BaseModel, Field


class SMA(BaseModel):
    time_period: int = Field(default=8)
    series_type: str = Field(default="close")


class EMA(BaseModel):
    time_period: int = Field(default=9)
    series_type: str = Field(default="close")


class MACD(BaseModel):
    series_type: str = Field(default="close")
    fastperiod: int = Field(default=12)
    slowperiod: int = Field(default=26)
    signalperiod: int = Field(default=9)


class BBANDS(BaseModel):
    time_period: int = Field(default=20)
    series_type: str = Field(default="close")
    dev: float = Field(default=1.5)


class RSI(BaseModel):
    series_type: str = Field(default="close")
    time_period: int = Field(default=14)


indicator_required_settings = {
    "SMA": SMA,
    "EMA": EMA,
    "MACD": MACD,
    "BBANDS": BBANDS,
    "RSI": RSI,
}
