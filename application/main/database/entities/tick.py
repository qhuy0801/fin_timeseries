from sqlalchemy import Integer, Column, DateTime, Float
from sqlalchemy.orm import declarative_base


def tick(Base, table_name):
    class Tick(Base):
        __tablename__ = f"{table_name}"
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, unique=True)
        open = Column(Float)
        high = Column(Float)
        low = Column(Float)
        close = Column(Float)
    return Tick
