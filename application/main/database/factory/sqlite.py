import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

engine = create_engine(os.environ["SQLITE_LOCATION"], echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)