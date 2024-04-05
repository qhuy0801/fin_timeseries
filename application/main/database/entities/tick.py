import warnings
from sqlalchemy import exc as sa_exc
from sqlalchemy import Integer, Column, DateTime, Float


def tick(Base, table_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=sa_exc.SAWarning)

        class Tick(Base):
            __tablename__ = f"{table_name}"
            id = Column(Integer, primary_key=True)
            timestamp = Column(DateTime, unique=True)
            open = Column(Float)
            high = Column(Float)
            low = Column(Float)
            close = Column(Float)
    return Tick
