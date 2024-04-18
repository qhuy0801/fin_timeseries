import warnings
from sqlalchemy import exc as sa_exc
from sqlalchemy import Integer, Column, DateTime, Float


def macd_tick(Base, table_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=sa_exc.SAWarning)

        class MacdTick(Base):
            __tablename__ = f"{table_name}"
            id = Column(Integer, primary_key=True)
            timestamp = Column(DateTime, unique=True)
            macd = Column(Float)
            macd_hist = Column(Float)
            macd_signal = Column(Float)
    return MacdTick


def bbands_tick(Base, table_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=sa_exc.SAWarning)

        class BBandsTick(Base):
            __tablename__ = f"{table_name}"
            id = Column(Integer, primary_key=True)
            timestamp = Column(DateTime, unique=True)
            real_lower_band = Column(Float)
            real_middle_band = Column(Float)
            real_upper_band = Column(Float)
    return BBandsTick


def rsi_tick(Base, table_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=sa_exc.SAWarning)

        class RSITick(Base):
            __tablename__ = f"{table_name}"
            id = Column(Integer, primary_key=True)
            timestamp = Column(DateTime, unique=True)
            rsi = Column(Float)
    return RSITick