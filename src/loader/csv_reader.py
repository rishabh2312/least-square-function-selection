"""
CSV data loading utilities.
"""

import pandas as pd
from pathlib import Path
from src.exceptions import DataLoadError


class DataLoader:
    """
    Base class for data loaders.

    Attributes:
        base_dir: Root directory for loading files.
    """

    def __init__(self, base_dir: Path):
        """
        Initialize the data loader.

        Args:
            base_dir: Root directory path.
        """
        self.base_dir = Path(base_dir)

    def load(self, source: str) -> pd.DataFrame:
        """
        Load data from source. Must be implemented by subclasses.

        Args:
            source: Data source identifier.

        Returns:
            DataFrame with loaded data.

        Raises:
            NotImplementedError: Always, since this is abstract.
        """
        raise NotImplementedError("Subclasses must implement load()")


class CSVReader(DataLoader):
    """
    Loads CSV files from the data directory.

    Inherits from DataLoader and implements CSV-specific loading.

    Attributes:
        base_dir: Root directory containing the data folder.
    """

    def __init__(self, base_dir: Path):
        """
        Initialize the CSV reader.

        Args:
            base_dir: Root directory containing the data folder.
        """
        super().__init__(base_dir)

    def load(self, source: str) -> pd.DataFrame:
        """
        Load a CSV file.

        Args:
            source: Name of the CSV file.

        Returns:
            DataFrame with the CSV contents.

        Raises:
            DataLoadError: If file not found, empty, or malformed.
        """
        return self.load_csv_data(source)

    def load_csv_data(self, file_name: str) -> pd.DataFrame:
        """
        Load a CSV file from the data folder.

        Args:
            file_name: Name of the CSV file (e.g. 'train.csv').

        Returns:
            DataFrame containing the CSV data.

        Raises:
            DataLoadError: If file not found, empty, or cannot be parsed.
        """
        file_path = self.base_dir / "data" / file_name
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise DataLoadError(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            raise DataLoadError(f"File is empty: {file_path}")
        except pd.errors.ParserError as e:
            raise DataLoadError(f"Failed to parse CSV file {file_path}: {e}")
