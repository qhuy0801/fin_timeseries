from fastapi import FastAPI, Query

from application.main.requests.indicators_request import load_indicator_ticks
from application.main.utils.indicator_utils import _indicator_required_settings

app = FastAPI()


# Crawling (loading service)
# Indicators
@app.post("/craw/indicators/BBANDS/")
async def craw_indicators(
    body: _indicator_required_settings.get("BBANDS"),
    symbol: str = Query("AAPL", description="Trading symbols such as AAPL, MSFT, etc."),
    interval: str = Query(
        "15min", description="1min, 5min, 15min, 30min, 60min, daily, weekly, monthly"
    ),
    start_month: str = Query("2010-01", description="The first month of data"),
):
    load_indicator_ticks(
        func="BBANDS",
        symbol=symbol,
        interval=interval,
        start_month=start_month,
        time_period=body.time_period,
        series_type=body.series_type,
        nbdevup=body.nbdevup,
        nbdevdn=body.nbdevdn,
    )
    return {"message": "BBANDS loaded/updated!"}


@app.post("/craw/indicators/MACD/")
async def craw_indicators(
    body: _indicator_required_settings.get("MACD"),
    symbol: str = Query("AAPL", description="Trading symbols such as AAPL, MSFT, etc."),
    interval: str = Query(
        "15min", description="1min, 5min, 15min, 30min, 60min, daily, weekly, monthly"
    ),
    start_month: str = Query("2010-01", description="The first month of data"),
):
    load_indicator_ticks(
        func="MACD",
        symbol=symbol,
        interval=interval,
        start_month=start_month,
        series_type=body.series_type,
        fastperiod=body.fastperiod,
        slowperiod=body.slowperiod,
        signalperiod=body.signalperiod,
    )
    return {"message": "MACD loaded/updated!"}
