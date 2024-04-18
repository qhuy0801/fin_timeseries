from pydantic import BaseModel, Field


class SMA(BaseModel):
    time_period: int
    series_type: str


class EMA(BaseModel):
    time_period: int
    series_type: str


class MACD(BaseModel):
    series_type: str = Field(default="close")
    fastperiod: int = Field(default=12)
    slowperiod: int = Field(default=26)
    signalperiod: int = Field(default=9)


class BBANDS(BaseModel):
    time_period: int = Field(default=20)
    series_type: str = Field(default="close")
    nbdevup: float = Field(default=1.5)
    nbdevdn: float = Field(default=1.5)


_indicator_required_settings = {
    "SMA": SMA,
    "EMA": EMA,
    "MACD": MACD,
    "BBANDS": BBANDS
}
