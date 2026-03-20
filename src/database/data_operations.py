"""
Database operations - loading CSVs and updating records.
"""

from src.database.table_creation import get_engine, create_tables, get_session
from src.database.table_creation import Training, IdealFunction, TestResult
from src.loader.csv_reader import CSVReader
from src.exceptions import DatabaseError, DataValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)

engine = get_engine("sqlite:///results.db")
create_tables(engine)
session = get_session(engine)


def publish_csvs_to_db(base_dir: Path):
    """
    Load all CSV files into the database.

    Loads train.csv, ideal.csv and test.csv from the data folder
    into their respective database tables.

    Args:
        base_dir: Root directory containing the data folder.

    Raises:
        DatabaseError: If database insertion fails.
    """
    loader = CSVReader(base_dir)

    try:
        load_training_function_to_db(loader)
        load_ideal_function_to_db(loader)
        load_test_data_to_db(loader)
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error while inserting data: {e}")
        raise DatabaseError(f"Failed to load data into database: {e}")
    finally:
        session.close()


def load_training_function_to_db(loader: CSVReader):
    """
    Load training data from train.csv into the database.

    Args:
        loader: CSVReader instance for loading files.

    Raises:
        DataValidationError: If required columns are missing.
    """
    training_df = loader.load_csv_data("train.csv")

    required_cols = ['x', 'y1', 'y2', 'y3', 'y4']
    missing = [c for c in required_cols if c not in training_df.columns]
    if missing:
        raise DataValidationError(f"Training CSV missing columns: {missing}")

    row_count = session.query(Training).count()
    if row_count == 0:
        train_records = df_to_mappings(training_df)
        session.bulk_insert_mappings(Training, train_records)
        session.commit()
        logger.info(f"Inserted {len(train_records)} training records into database.")
    else:
        logger.info(f"Training table already contains {row_count} rows.")


def load_ideal_function_to_db(loader: CSVReader):
    """
    Load ideal function data from ideal.csv into the database.

    Args:
        loader: CSVReader instance for loading files.

    Raises:
        DataValidationError: If required columns (x, y1-y50) are missing.
    """
    ideal_df = loader.load_csv_data("ideal.csv")

    ideal_cols_needed = ['x'] + [f'y{i}' for i in range(1, 51)]
    missing = [c for c in ideal_cols_needed if c not in ideal_df.columns]
    if missing:
        raise DataValidationError(f"Ideal CSV missing columns: {missing}")

    row_count = session.query(IdealFunction).count()
    if row_count == 0:
        ideal_records = df_to_mappings(ideal_df[ideal_cols_needed])
        session.bulk_insert_mappings(IdealFunction, ideal_records)
        session.commit()
        logger.info(f"Inserted {len(ideal_records)} ideal function records into database.")
    else:
        logger.info(f"Ideal functions table already contains {row_count} rows.")


def load_test_data_to_db(loader: CSVReader):
    """
    Load test data from test.csv into the database.

    Only loads x and y columns. The delta_y and ideal_no fields
    are populated later by the mapping process.

    Args:
        loader: CSVReader instance for loading files.

    Raises:
        DataValidationError: If x or y columns are missing.
    """
    test_df = loader.load_csv_data("test.csv")

    if 'x' not in test_df.columns or 'y' not in test_df.columns:
        raise DataValidationError("Test CSV must have 'x' and 'y' columns")

    row_count = session.query(TestResult).count()
    if row_count == 0:
        test_records = test_df[['x', 'y']].to_dict(orient='records')
        session.bulk_insert_mappings(TestResult, test_records)
        session.commit()
        logger.info(f"Inserted {len(test_records)} test records into database.")
    else:
        logger.info(f"Test results table already contains {row_count} rows.")


def read_data(model):
    """
    Read all rows from a database table.

    Args:
        model: SQLAlchemy model class (Training, IdealFunction, or TestResult).

    Returns:
        List of ORM objects ordered by x value.
    """
    return session.query(model).order_by(model.x).all()


def persist_mapped_test_results_update(mapped_df: pd.DataFrame, table_name: str = "test_results"):
    """
    Update the test results table with mapping results.

    Updates delta_y and ideal_no columns based on the mapping
    performed by TestPointMapper.

    Args:
        mapped_df: DataFrame with x, y, delta_y, ideal_function columns.
        table_name: Name of the table to update.

    Raises:
        DatabaseError: If the update fails.
    """
    conn = engine.connect()
    trans = conn.begin()

    try:
        for _, row in mapped_df.iterrows():
            x = float(row["x"])
            y = float(row["y"])
            delta = row["delta_y"]
            ideal_no = row["ideal_function"]

            if pd.isna(delta) or ideal_no is None:
                conn.execute(
                    text(f"UPDATE {table_name} SET delta_y = NULL, ideal_no = NULL "
                         f"WHERE x = :x AND y = :y"),
                    {"x": x, "y": y}
                )
            else:
                conn.execute(
                    text(f"UPDATE {table_name} SET delta_y = :delta, ideal_no = :ideal "
                         f"WHERE x = :x AND y = :y"),
                    {"delta": float(delta), "ideal": int(ideal_no), "x": x, "y": y}
                )

        trans.commit()
        logger.info(f"Updated {table_name} with mapping results.")

    except SQLAlchemyError as e:
        trans.rollback()
        raise DatabaseError(f"Failed to update test results: {e}")
    finally:
        conn.close()


def df_to_mappings(df: pd.DataFrame, column_map: dict = None) -> list:
    """
    Convert a DataFrame to a list of dicts for bulk insert.

    Args:
        df: DataFrame to convert.
        column_map: Optional dict to rename columns before conversion.

    Returns:
        List of dicts suitable for bulk_insert_mappings.
    """
    if column_map is None:
        return df.to_dict(orient="records")

    df_renamed = df.rename(columns=column_map)
    return df_renamed.to_dict(orient="records")
