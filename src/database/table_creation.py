"""
SQLAlchemy ORM models and database setup.
"""

from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import logging

Base = declarative_base()
logger = logging.getLogger(__name__)


class Training(Base):
    """
    ORM model for training function data.

    Stores four training functions (y1-y4) for each x value.

    Attributes:
        id: Primary key.
        x: X coordinate value.
        y1, y2, y3, y4: Y values for each training function.
    """
    __tablename__ = "training_functions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False)
    y1 = Column(Float, nullable=True)
    y2 = Column(Float, nullable=True)
    y3 = Column(Float, nullable=True)
    y4 = Column(Float, nullable=True)


class TestResult(Base):
    """
    ORM model for test data and mapping results.

    Initially stores just x and y from test.csv. The delta_y and
    ideal_no fields are populated after mapping.

    Attributes:
        id: Primary key.
        x: X coordinate of test point.
        y: Y value of test point.
        delta_y: Deviation from matched ideal function (None if unassigned).
        ideal_no: Number of matched ideal function (None if unassigned).
    """
    __tablename__ = 'test_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    delta_y = Column(Float, nullable=True)
    ideal_no = Column(Integer, nullable=True)


def _make_ideal_class():
    """
    Dynamically create the IdealFunction class with 50 y-columns.

    Returns:
        IdealFunction class with x and y1-y50 columns.
    """
    attrs = {
        '__tablename__': 'ideal_functions',
        '__doc__': "ORM model for ideal function data. Stores 50 ideal functions (y1-y50) for each x value.",
        'id': Column(Integer, primary_key=True, autoincrement=True),
        'x': Column(Float, nullable=False)
    }
    for i in range(1, 51):
        attrs[f'y{i}'] = Column(Float, nullable=True)

    return type('IdealFunction', (Base,), attrs)


IdealFunction = _make_ideal_class()


def get_engine(db_path: str = "sqlite:///results.db", echo: bool = False):
    """
    Create a SQLAlchemy database engine.

    Args:
        db_path: Database connection string.
        echo: If True, log all SQL statements.

    Returns:
        SQLAlchemy Engine instance.
    """
    return create_engine(db_path, echo=echo, future=True)


def create_tables(engine):
    """
    Create all database tables if they don't exist.

    Args:
        engine: SQLAlchemy Engine instance.
    """
    Base.metadata.create_all(bind=engine)


def get_session(engine):
    """
    Create a new database session.

    Args:
        engine: SQLAlchemy Engine instance.

    Returns:
        New Session instance.
    """
    session_factory = sessionmaker(bind=engine)
    return session_factory()
