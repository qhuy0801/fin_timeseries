import os
from typing import Optional, List

import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from sklearn.preprocessing import minmax_scale
from sqlalchemy import create_engine, or_, and_, desc, func

from sqlalchemy.orm import sessionmaker, declarative_base

from application.main.algo.dynamic import dynamic_processing
from application.main.database.entities.correlation import Correlation
from application.main.database.entities.stock_tick import stock_tick
from application.main.requests.alpha_vantage import get_table_name
from application.main.utils.timeframe_processing import timeframe_cropping

load_dotenv()

engine = create_engine(os.environ["SQLITE_LOCATION"], echo=False)
Session = sessionmaker(bind=engine, expire_on_commit=True)

Base = declarative_base()


def generate_correlation(
    func: str,
    interval: Optional[str],
    symbols: List[str],
    target_field: Optional[str] = "close",
):
    """
    Get the list of symbols, process the data and calculate Pearson correlation
    Save to correlation table
    Args:
        func:
        interval:
        symbols:
        target_field:

    Returns: None

    """
    # Get table list by finding the table names
    table_list = [get_table_name(func, interval, symbol) for symbol in symbols]

    # Tick list
    table_list = {stock_tick(Base, name): name for name in table_list}

    correlation_generator = dynamic_processing(
        list(table_list.keys()), query_and_compute, target_field=target_field
    )

    with Session() as session:
        try:
            for table_1, table_2, (corr, p) in correlation_generator:
                new_correlation = Correlation(
                    symbol_1=table_list.get(table_1),
                    symbol_2=table_list.get(table_2),
                    correlation=corr,
                    p_value=p,
                    target_field=str(target_field),
                )
                session.add(new_correlation)
            session.commit()
        except Exception as e:
            print(f"An error occurred: {e}")
            session.rollback()
        finally:
            session.close()


def query_and_compute(tick_1, tick_2, target_field: str = "close") -> (float, float):
    """
    Get table name, match timeframe, scale data, calculate Pearson correlation based on target field
    Args:
        tick_1:
        tick_2:
        target_field:

    Returns:
        scipy.stats._result_classes.PearsonRResult -> (float, float)
    """
    with Session() as session:
        # Check table
        if not hasattr(tick_1, target_field):
            raise AttributeError(f"{target_field} is not a valid attribute of tick_1.")
        if not hasattr(tick_2, target_field):
            raise AttributeError(f"{target_field} is not a valid attribute of tick_2.")

        # Query
        query_1 = session.query(tick_1).order_by(tick_1.timestamp.asc())
        query_2 = session.query(tick_2).order_by(tick_2.timestamp.asc())

        # Convert to pandas and crop the timeframe
        table_1, table_2 = timeframe_cropping(
            pd.read_sql(query_1.statement, session.bind),
            pd.read_sql(query_2.statement, session.bind),
        )

        # Scale data
        table_1, table_2 = minmax_scale(table_1[target_field].to_numpy()), minmax_scale(
            table_2[target_field].to_numpy()
        )

        return pearsonr(table_1, table_2)


def query_and_filter(
    symbol: Optional[str] = "QCOM",
    function: Optional[str] = "TIME_SERIES_INTRADAY",
    interval: Optional[str] = "15min",
    sort: Optional[str] = "abs_desc",
    limit: Optional[int] = 10,
):
    """
    Queries records from the Correlation table based on optional filters for symbols, function, and interval.

    Args:
        symbol (Optional[str]): Stock symbols to filter the data, or None to fetch all records.
        function (Optional[str]): The function type to filter by (default "TIME_SERIES_INTRADAY").
        interval (Optional[str]): The interval to filter by (default "15min").
        sort (Optional[str]): Sorting method ('asc', 'desc', 'abs_asc', 'abs_desc').
        limit: (Optional[int]): The number of records to return (default 10).

    Returns:
        pd.DataFrame: A DataFrame containing the filtered records.
    """
    with Session() as session:
        query = session.query(Correlation)

        # Build the LIKE pattern based on optional function and interval
        if function and interval:
            symbol_pattern = f"{function}_{interval}_{symbol}"
        else:
            # If function and interval aren't specified, just search for the symbol
            symbol_pattern = f"%{symbol}"

        # Apply the filter condition
        query = query.filter(
            or_(
                Correlation.symbol_1.like(symbol_pattern),
                Correlation.symbol_2.like(symbol_pattern),
            )
        )

        # Apply sorting based on the sort argument
        if sort == "asc":
            query = query.order_by(Correlation.correlation)
        elif sort == "desc":
            query = query.order_by(desc(Correlation.correlation))
        elif sort == "abs_asc":
            query = query.order_by(func.abs(Correlation.correlation))
        elif sort == "abs_desc":
            query = query.order_by(desc(func.abs(Correlation.correlation)))

        # Apply limit to the query results if showing is specified
        if limit is not None and isinstance(limit, int):
            query = query.limit(limit)

        df = pd.read_sql(query.statement, session.bind)
        return df


if __name__ == "__main__":
    df = query_and_filter("QCOM", function=None, interval=None, sort="asc")
    print(df)
