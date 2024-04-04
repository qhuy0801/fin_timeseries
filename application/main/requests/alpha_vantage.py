import gc
import os
import time
from datetime import datetime
from typing import List, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker, declarative_base

from application.main.database.entities.tick import tick
from application.main.requests.rest_request import fetch_data


load_dotenv()

engine = create_engine(os.environ["SQLITE_LOCATION"], echo=False)
Session = sessionmaker(bind=engine, expire_on_commit=True)

Base = declarative_base()


def load_ticks(
    func: str,
    interval: str,
    symbol: str,
    url: str = os.environ["ALPHA_VANTAGE"],
    method: str = "GET",
    output_size: str = "full",
    datatype: str = "csv",
    apikey: str = os.environ["ALPHA_VANTAGE_KEY"],
    extended_hours: bool = False,
    start_month: str = "2000-01",
    reverse_data: bool = True,
    sleep: int = 1,
):
    months = month_list(start_month)

    table_name = param_check(
        func=func, interval=interval, symbol=symbol
    )

    _tick, _last_timestamp_ = last_timestamp(table_name)

    if _last_timestamp_:
        for index, month in enumerate(months):
            if datetime.strptime(month, "%Y-%m") >= datetime(
                _last_timestamp_.year, _last_timestamp_.month, 1
            ):
                months = months[index:]

    for month in months:
        params = {
            "function": func,
            "symbol": symbol,
            "outputsize": output_size,
            "datatype": datatype,
            "interval": interval,
            "apikey": apikey,
            "extended_hours": "false" if not extended_hours else "true",
            "month": month,
        }
        response = fetch_data(
            url=url,
            params=params,
            method=method,
            reverse_data=reverse_data,
        )
        time.sleep(sleep)
        if isinstance(response, pd.DataFrame):
            response["timestamp"] = pd.to_datetime(response["timestamp"])
            if response.empty:
                continue
            if response["timestamp"].iat[0].replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            ) > datetime.strptime(month, "%Y-%m"):
                continue
            else:
                if _last_timestamp_:
                    response = response[response["timestamp"] > _last_timestamp_]
                with Session() as session:
                    for index, row in response.iterrows():
                        _tick_ = _tick(
                            timestamp=pd.to_datetime(row["timestamp"]),
                            open=row["open"],
                            high=row["high"],
                            low=row["low"],
                            close=row["close"],
                        )
                        session.add(_tick_)
                    try:
                        session.commit()
                    except IntegrityError:
                        break
    session.close()
    gc.collect()
    return None


def last_timestamp(table_name: str) -> tuple[Any, None] | None | Any:
    if not inspect(engine).has_table(table_name):
        _tick = tick(Base, table_name)
        Base.metadata.create_all(engine)
        return _tick, None
    else:
        with Session() as session:
            _tick = tick(Base, table_name)
            timestamp = (
                session.query(_tick.timestamp)
                .order_by(_tick.timestamp.desc())
                .first()
                .timestamp
            )
        return _tick, timestamp if timestamp else None


def month_list(start_month: str) -> List[str]:
    return (
        pd.date_range(
            start=start_month, end=datetime.now().strftime("%Y-%m"), freq="MS"
        )
        .strftime("%Y-%m")
        .tolist()
    )


def param_check(func: str, interval: Optional[str], symbol: str) -> str:
    if interval in ["daily, weekly, monthly", None]:
        return "_".join([func, symbol])
    else:
        return "_".join([func, interval, symbol])
