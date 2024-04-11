from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Correlation(Base):
    __tablename__ = 'correlations'
    id = Column(Integer, primary_key=True)
    symbol_1 = Column(String(10), nullable=False)
    symbol_2 = Column(String(10), nullable=False)
    correlation = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    target_field = Column(String(10))
