from sqlalchemy import Boolean, Column, Integer, String, Text
from database import Base


class History(Base):
    __tablename__ = "eegsession"  # Table name in the database

    id = Column(Integer, primary_key=True, index=True, name="SessionID")
    timestamp = Column(Text, nullable=False, name="Timestamp")  # Assuming timestamp is stored as text
    duration = Column(Text, nullable=False, name="Duration")  # Assuming duration is stored as text
    result = Column(Text, nullable=False, name="Result")  # Assuming result is stored as text
