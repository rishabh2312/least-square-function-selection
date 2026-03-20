"""
Core algorithm for selecting ideal functions and mapping test points.
"""

import logging
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Optional, List
from math import sqrt

from src.exceptions import DataValidationError, MappingError

logger = logging.getLogger(__name__)


class FunctionAnalyzer:
    """
    Base class for function analysis with shared validation.

    Provides common validation logic for training and ideal function data.

    Attributes:
        training_rows: List of training data ORM objects.
        ideal_rows: List of ideal function ORM objects.
    """

    def __init__(self, training_rows: Sequence, ideal_rows: Sequence):
        """
        Initialize with training and ideal data.

        Args:
            training_rows: Sequence of Training ORM objects.
            ideal_rows: Sequence of IdealFunction ORM objects.

        Raises:
            DataValidationError: If either dataset is empty.
        """
        if not training_rows or not ideal_rows:
            raise DataValidationError("Training and ideal data must not be empty")

        self.training_rows = list(training_rows)
        self.ideal_rows = list(ideal_rows)
        self._validate_data()

    def _validate_data(self):
        """
        Validate that x values match between training and ideal data.

        Raises:
            DataValidationError: If x coordinates don't match.
        """
        x_train = np.array([r.x for r in self.training_rows], dtype=float)
        x_ideal = np.array([r.x for r in self.ideal_rows], dtype=float)

        if not np.allclose(x_train, x_ideal):
            raise DataValidationError(
                "X values differ between training and ideal data. "
                "Ensure both datasets have the same x-coordinates."
            )


class IdealFunctionSelector(FunctionAnalyzer):
    """
    Selects best ideal function for each training function using least squares.

    Inherits from FunctionAnalyzer for data validation.

    Attributes:
        TRAIN_COLS: Column names for training data (y1-y4).
        IDEAL_COLS: Column names for ideal functions (y1-y50).
        results: DataFrame with selection results after calling select().
    """

    TRAIN_COLS = ["y1", "y2", "y3", "y4"]
    IDEAL_COLS = [f"y{i}" for i in range(1, 51)]

    def __init__(self, training_rows: Sequence, ideal_rows: Sequence):
        """
        Initialize selector with training and ideal data.

        Args:
            training_rows: Sequence of Training ORM objects.
            ideal_rows: Sequence of IdealFunction ORM objects.

        Raises:
            DataValidationError: If data is empty or x values don't match.
        """
        super().__init__(training_rows, ideal_rows)
        self.results = None
        self._x, self._train_matrix, self._ideal_matrix = self._build_matrices()

    def _build_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert ORM rows to numpy matrices for vectorized computation.

        Returns:
            Tuple of (x_values, training_matrix, ideal_matrix).
        """
        x = np.array([r.x for r in self.training_rows], dtype=float)

        train_matrix = np.column_stack([
            [float(getattr(r, c)) for r in self.training_rows]
            for c in self.TRAIN_COLS
        ])

        ideal_matrix = np.column_stack([
            [float(getattr(r, c)) for r in self.ideal_rows]
            for c in self.IDEAL_COLS
        ])

        return x, train_matrix, ideal_matrix

    def select(self) -> pd.DataFrame:
        """
        Find best ideal function for each training function.

        Uses least squares (sum of squared deviations) to find the
        ideal function that best matches each training function.

        Returns:
            DataFrame with columns: training_col, best_ideal_fun,
            best_min_ssd, max_dev.

        Raises:
            DataValidationError: If training data contains non-finite values.
        """
        results = []

        for idx in range(self._train_matrix.shape[1]):
            train_vec = self._train_matrix[:, idx]

            if not np.all(np.isfinite(train_vec)):
                raise DataValidationError(
                    f"Non-finite values found in training column y{idx + 1}"
                )

            ssds = self._compute_ssd(train_vec, self._ideal_matrix)
            best_idx = int(np.argmin(ssds))
            best_ssd = float(ssds[best_idx])
            best_ideal_vec = self._ideal_matrix[:, best_idx]
            max_dev = float(np.max(np.abs(train_vec - best_ideal_vec)))

            results.append({
                "training_col": f"y{idx + 1}",
                "best_ideal_fun": best_idx + 1,
                "best_min_ssd": best_ssd,
                "max_dev": max_dev
            })

        self.results = pd.DataFrame(results)

        # Log selection results in a clear, readable format
        logger.info("=" * 80)
        logger.info("IDEAL FUNCTION SELECTION COMPLETE")
        logger.info("=" * 80)
        for result in results:
            logger.info(
                f"Training {result['training_col'].upper()} → "
                f"Ideal Function {result['best_ideal_fun']:2d} | "
                f"SSD: {result['best_min_ssd']:10.4f} | "
                f"Max Deviation: {result['max_dev']:.6f}"
            )
        logger.info("=" * 80)

        return self.results

    def _compute_ssd(self, train_vec: np.ndarray, ideal_matrix: np.ndarray) -> np.ndarray:
        """
        Compute sum of squared deviations between training vector and all ideals.

        Args:
            train_vec: Training function y-values.
            ideal_matrix: Matrix of all ideal function y-values.

        Returns:
            Array of SSD values, one per ideal function.
        """
        diff = ideal_matrix - train_vec[:, None]
        return np.einsum('ij,ij->j', diff, diff)


class TestPointMapper:
    """
    Maps test points to selected ideal functions.

    Uses threshold of sqrt(2) * max_deviation to determine if a test
    point can be assigned to an ideal function.

    Attributes:
        test_rows: List of test point ORM objects.
        ideal_rows: List of ideal function ORM objects.
        results_df: DataFrame with selection results from IdealFunctionSelector.
    """

    def __init__(self, test_rows: Sequence, ideal_rows: Sequence, results_df: pd.DataFrame):
        """
        Initialize mapper with test data and selection results.

        Args:
            test_rows: Sequence of TestResult ORM objects.
            ideal_rows: Sequence of IdealFunction ORM objects.
            results_df: DataFrame from IdealFunctionSelector.select().

        Raises:
            MappingError: If results_df is missing required columns.
        """
        required_cols = ['training_col', 'best_ideal_fun', 'best_min_ssd', 'max_dev']
        missing = [c for c in required_cols if c not in results_df.columns]
        if missing:
            raise MappingError(f"Results DataFrame missing columns: {missing}")

        self.test_rows = list(test_rows)
        self.ideal_rows = list(ideal_rows)
        self.results_df = results_df
        self._ideal_x = np.array([r.x for r in ideal_rows], dtype=float)

    def map(self) -> pd.DataFrame:
        """
        Map each test point to a selected ideal function.

        A test point is assigned to an ideal function if its deviation
        is within sqrt(2) * max_deviation threshold.

        Returns:
            DataFrame with columns: x, y, delta_y, ideal_function.
            Points not matching any ideal have None for delta_y and ideal_function.
        """
        selected_ideals = self._prepare_selected_ideals()
        mapped = []

        for test in self.test_rows:
            x_t, y_t = float(test.x), float(test.y)
            best_ideal, best_delta = None, np.inf

            for ideal in selected_ideals:
                y_ideal = np.interp(x_t, self._ideal_x, ideal["y_values"])
                delta = abs(y_t - y_ideal)

                if delta <= ideal["threshold"] and delta < best_delta:
                    best_delta = delta
                    best_ideal = ideal["ideal_no"]

            mapped.append({
                "x": x_t,
                "y": y_t,
                "delta_y": float(best_delta) if best_ideal else None,
                "ideal_function": best_ideal
            })

        return pd.DataFrame(mapped)

    def _prepare_selected_ideals(self) -> List[dict]:
        """
        Prepare selected ideal functions with their thresholds.

        Returns:
            List of dicts with ideal_no, y_values, and threshold for each
            selected ideal function.
        """
        selected = []
        for _, row in self.results_df.iterrows():
            ideal_no = int(row["best_ideal_fun"])
            max_dev = float(row["max_dev"])

            y_values = np.array([
                getattr(r, f"y{ideal_no}") for r in self.ideal_rows
            ], dtype=float)

            selected.append({
                "ideal_no": ideal_no,
                "y_values": y_values,
                "threshold": max_dev * sqrt(2)
            })
        return selected


def least_squares_select(training_data, ideal_data) -> pd.DataFrame:
    """
    Select best ideal function for each training function.

    Args:
        training_data: Sequence of Training ORM objects.
        ideal_data: Sequence of IdealFunction ORM objects.

    Returns:
        DataFrame with selection results.
    """
    selector = IdealFunctionSelector(training_data, ideal_data)
    return selector.select()


def map_test_points_from_results(test_rows, ideal_rows, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map test points to selected ideal functions.

    Args:
        test_rows: Sequence of TestResult ORM objects.
        ideal_rows: Sequence of IdealFunction ORM objects.
        results_df: DataFrame from least_squares_select().

    Returns:
        DataFrame with mapping results.
    """
    mapper = TestPointMapper(test_rows, ideal_rows, results_df)
    return mapper.map()


def rows_to_matrices(training_rows: Sequence, ideal_rows: Sequence,
                     train_cols: Optional[Sequence[str]] = None,
                     ideal_cols: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ORM rows to numpy matrices for computation.

    Args:
        training_rows: Sequence of Training ORM objects.
        ideal_rows: Sequence of IdealFunction ORM objects.
        train_cols: Column names for training data. Defaults to y1-y4.
        ideal_cols: Column names for ideal data. Defaults to y1-y50.

    Returns:
        Tuple of (x_values, training_matrix, ideal_matrix).

    Raises:
        ValueError: If rows are empty or x values don't match.
    """
    if train_cols is None:
        train_cols = ["y1", "y2", "y3", "y4"]
    if ideal_cols is None:
        ideal_cols = [f"y{i}" for i in range(1, 51)]

    training_rows = list(training_rows)
    ideal_rows = list(ideal_rows)

    if not training_rows or not ideal_rows:
        raise ValueError("training_rows and ideal_rows must be non-empty")

    x_train = np.array([r.x for r in training_rows], dtype=float)
    x_ideal = np.array([r.x for r in ideal_rows], dtype=float)

    if not np.allclose(x_train, x_ideal):
        raise ValueError("x grids differ between training and ideal rows")

    train_matrix = np.column_stack([
        [float(getattr(r, c)) for r in training_rows] for c in train_cols
    ])
    ideal_matrix = np.column_stack([
        [float(getattr(r, c)) for r in ideal_rows] for c in ideal_cols
    ])

    return x_train, train_matrix, ideal_matrix


def compute_ssd_vectorized(train_vec: np.ndarray, ideal_matrix: np.ndarray) -> np.ndarray:
    """
    Compute sum of squared deviations between training vector and all ideals.

    Args:
        train_vec: Training function y-values as 1D array.
        ideal_matrix: Matrix of ideal function y-values.

    Returns:
        Array of SSD values, one per ideal function column.
    """
    diff = ideal_matrix - train_vec[:, None]
    return np.einsum('ij,ij->j', diff, diff)


def summary_print(mapped_df: pd.DataFrame):
    """
    Log a summary of the mapping results.

    Args:
        mapped_df: DataFrame from map_test_points_from_results().
    """
    total = len(mapped_df)
    assigned = mapped_df["ideal_function"].notna().sum()
    unassigned = total - assigned
    assignment_rate = (assigned / total * 100) if total > 0 else 0

    logger.info("=" * 80)
    logger.info("TEST POINT MAPPING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Test Points:     {total:3d}")
    logger.info(f"Successfully Assigned: {assigned:3d} ({assignment_rate:5.1f}%)")
    logger.info(f"Unassigned (Rejected): {unassigned:3d} ({100-assignment_rate:5.1f}%)")
    logger.info("-" * 80)

    # Show distribution across ideal functions
    if assigned > 0:
        logger.info("Distribution of Assigned Points:")
        assignment_counts = mapped_df[mapped_df["ideal_function"].notna()].groupby("ideal_function").size()
        for ideal_no, count in assignment_counts.items():
            percentage = (count / assigned * 100)
            logger.info(f"  Ideal Function {int(ideal_no):2d}: {count:3d} points ({percentage:5.1f}% of assigned)")

    logger.info("=" * 80)
