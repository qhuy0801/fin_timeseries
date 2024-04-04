from sqlalchemy import Column, Integer, String, Text, Float
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Company(Base):
    __tablename__ = 'companies'
    id = Column(Integer, primary_key=True)
    exchange = Column(String)
    symbol = Column(String, unique=True)
    shortname = Column(String)
    longname = Column(String)
    sector = Column(String)
    industry = Column(String)
    city = Column(String)
    state = Column(String)
    country = Column(String)
    business_summary = Column(Text)
    weight = Column(Float)
