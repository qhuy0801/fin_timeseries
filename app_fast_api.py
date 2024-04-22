import os

import uvicorn

import wandb
from dotenv import load_dotenv
from fastapi import FastAPI, Query

from application.main.requests.indicators_request import load_indicator_ticks
from application.main.requests.stock_request import load_stock_ticks
from application.main.utils import indicator_required_settings
# from application.main.utils import SweepConfig

load_dotenv()
wandb.login(key=os.environ["WANDB_KEY"])

app = FastAPI()


# Crawling (loading service) / Stock timeseries
@app.get("/craw/stocks/")
async def craw_stock(
    func: str = Query(
        "TIME_SERIES_INTRADAY",
        description="TIME_SERIES_INTRADAY, TIME_SERIES_DAILY, TIME_SERIES_DAILY_ADJUSTED",
    ),
    interval: str = Query(
        "15min", description="1min, 5min, 15min, 30min, 60min for TIME_SERIES_INTRADAY"
    ),
    symbol: str = Query("QCOM", description="The asset symbol"),
    adjusted: bool = Query(
        False,
        description="True for output time series is adjusted by historical split and dividend events, "
                    "False to query raw (as-traded) intraday values",
    ),
    extended_hours: bool = Query(
        True,
        description="True - the output time series will include both the regular trading hours and the "
                    "extended trading hours (4:00am to 8:00pm Eastern Time for the US market)",
    ),
    start_month: str = Query(
        "2010-01", description="The the initial month for data crawling"
    ),
):
    load_stock_ticks(
        func=func,
        interval=interval,
        symbol=symbol,
        start_month=start_month,
        adjusted=adjusted,
        extended_hours=extended_hours,
    )
    return {"message": f"Price data for {symbol} for {func} {interval} timeframe has been updated"}


# Crawling (loading service) / Indicators
@app.post("/craw/indicators/BBANDS/")
async def craw_indicators(
    body: indicator_required_settings.get("BBANDS"),
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
    body: indicator_required_settings.get("MACD"),
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


# Logging services
# Create sweep (Hyper-parameter tuning logs)
# @app.post("/sweep/create/")
# async def create_sweep(sweep_config: SweepConfig):
#     return {"message": f"Sweep created {wandb.sweep(sweep_config.dict(), project=sweep_config.project)}!"}


# For debugging
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
