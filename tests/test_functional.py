import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

from src.loader.csv_reader import CSVReader
from src.exceptions import DataLoadError
from src.service.ideal_function_selector import (
    least_squares_select,
    map_test_points_from_results,
    rows_to_matrices,
    compute_ssd_vectorized,
    summary_print
)


class TestCSVReader:

    def test_load_training_data(self, temp_data_dir):
        reader = CSVReader(temp_data_dir)
        df = reader.load_csv_data("train.csv")

        assert isinstance(df, pd.DataFrame)
        assert 'x' in df.columns
        assert all(f'y{i}' in df.columns for i in range(1, 5))

    def test_load_ideal_data(self, temp_data_dir):
        reader = CSVReader(temp_data_dir)
        df = reader.load_csv_data("ideal.csv")

        assert 'x' in df.columns
        assert all(f'y{i}' in df.columns for i in range(1, 51))

    def test_load_test_data(self, temp_data_dir):
        reader = CSVReader(temp_data_dir)
        df = reader.load_csv_data("test.csv")

        assert 'x' in df.columns
        assert 'y' in df.columns

    def test_file_not_found(self, temp_data_dir):
        reader = CSVReader(temp_data_dir)
        with pytest.raises(DataLoadError):
            reader.load_csv_data("nonexistent.csv")


class TestRowsToMatrices:

    def test_returns_correct_shapes(self, mock_training_rows, mock_ideal_rows):
        x, train_matrix, ideal_matrix = rows_to_matrices(mock_training_rows, mock_ideal_rows)

        assert x.shape == (len(mock_training_rows),)
        assert train_matrix.shape == (len(mock_training_rows), 4)
        assert ideal_matrix.shape == (len(mock_ideal_rows), 50)

    def test_empty_rows_raises_error(self):
        with pytest.raises(ValueError):
            rows_to_matrices([], [])


class TestComputeSSD:

    def test_identical_vectors_zero_ssd(self):
        vec = np.array([1.0, 2.0, 3.0])
        matrix = vec.reshape(-1, 1)
        result = compute_ssd_vectorized(vec, matrix)
        assert result[0] == 0.0

    def test_ssd_calculation(self):
        train = np.array([1.0, 2.0, 3.0])
        ideal = np.array([2.0, 3.0, 4.0])
        result = compute_ssd_vectorized(train, ideal.reshape(-1, 1))
        # (2-1)^2 + (3-2)^2 + (4-3)^2 = 3
        assert result[0] == 3.0


class TestLeastSquaresSelect:

    def test_returns_dataframe_with_correct_columns(self, mock_training_rows, mock_ideal_rows):
        result = least_squares_select(mock_training_rows, mock_ideal_rows)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert all(col in result.columns for col in
                   ['training_col', 'best_ideal_fun', 'best_min_ssd', 'max_dev'])

    def test_selects_exact_match(self, mock_training_rows, mock_ideal_rows):
        result = least_squares_select(mock_training_rows, mock_ideal_rows)

        assert result.iloc[0]['best_ideal_fun'] == 1
        assert result.iloc[1]['best_ideal_fun'] == 2
        assert result.iloc[2]['best_ideal_fun'] == 3
        assert result.iloc[3]['best_ideal_fun'] == 4

    def test_exact_match_has_zero_ssd(self, mock_training_rows, mock_ideal_rows):
        result = least_squares_select(mock_training_rows, mock_ideal_rows)

        for ssd in result['best_min_ssd']:
            assert ssd < 1e-10

    def test_best_ideal_in_valid_range(self, mock_training_rows, mock_ideal_rows):
        result = least_squares_select(mock_training_rows, mock_ideal_rows)

        for ideal in result['best_ideal_fun']:
            assert 1 <= ideal <= 50


class TestMapTestPoints:

    def test_returns_dataframe_with_correct_columns(self, mock_test_rows, mock_ideal_rows):
        results_df = pd.DataFrame({
            'training_col': ['y1', 'y2', 'y3', 'y4'],
            'best_ideal_fun': [1, 2, 3, 4],
            'best_min_ssd': [0.0, 0.0, 0.0, 0.0],
            'max_dev': [1.0, 1.0, 1.0, 1.0]
        })

        mapped = map_test_points_from_results(mock_test_rows, mock_ideal_rows, results_df)

        assert isinstance(mapped, pd.DataFrame)
        assert all(col in mapped.columns for col in ['x', 'y', 'delta_y', 'ideal_function'])

    def test_preserves_test_point_count(self, mock_test_rows, mock_ideal_rows):
        results_df = pd.DataFrame({
            'training_col': ['y1', 'y2', 'y3', 'y4'],
            'best_ideal_fun': [1, 2, 3, 4],
            'best_min_ssd': [0.0, 0.0, 0.0, 0.0],
            'max_dev': [10.0, 10.0, 10.0, 10.0]
        })

        mapped = map_test_points_from_results(mock_test_rows, mock_ideal_rows, results_df)
        assert len(mapped) == len(mock_test_rows)

    def test_assigns_points_within_threshold(self, mock_test_rows, mock_ideal_rows):
        results_df = pd.DataFrame({
            'training_col': ['y1', 'y2', 'y3', 'y4'],
            'best_ideal_fun': [1, 2, 3, 4],
            'best_min_ssd': [0.0, 0.0, 0.0, 0.0],
            'max_dev': [10.0, 10.0, 10.0, 10.0]
        })

        mapped = map_test_points_from_results(mock_test_rows, mock_ideal_rows, results_df)
        assigned = mapped['ideal_function'].notna().sum()
        assert assigned > 0

    def test_unassigned_points_have_null_values(self):
        from tests.conftest import MockRow

        x_vals = np.linspace(-10, 10, 50)
        ideal_rows = []
        for x in x_vals:
            data = {'x': x, 'y1': x}
            for j in range(2, 51):
                data[f'y{j}'] = x * 1000
            ideal_rows.append(MockRow(**data))

        results_df = pd.DataFrame({
            'training_col': ['y1', 'y2', 'y3', 'y4'],
            'best_ideal_fun': [1, 1, 1, 1],
            'best_min_ssd': [0.0, 0.0, 0.0, 0.0],
            'max_dev': [0.1, 0.1, 0.1, 0.1]
        })

        test_rows = [MockRow(x=5.0, y=100.0)]
        mapped = map_test_points_from_results(test_rows, ideal_rows, results_df)

        assert pd.isna(mapped.iloc[0]['ideal_function'])
        assert pd.isna(mapped.iloc[0]['delta_y'])


class TestIntegration:

    def test_full_pipeline(self, temp_data_dir):
        from tests.conftest import MockRow

        reader = CSVReader(temp_data_dir)
        training_df = reader.load_csv_data("train.csv")
        ideal_df = reader.load_csv_data("ideal.csv")
        test_df = reader.load_csv_data("test.csv")

        assert len(training_df) > 0
        assert len(ideal_df) > 0
        assert len(test_df) > 0

        training_rows = [MockRow(**row.to_dict()) for _, row in training_df.iterrows()]
        ideal_rows = [MockRow(**row.to_dict()) for _, row in ideal_df.iterrows()]
        test_rows = [MockRow(x=row['x'], y=row['y']) for _, row in test_df.iterrows()]

        results_df = least_squares_select(training_rows, ideal_rows)
        assert len(results_df) == 4
        assert all(1 <= ideal <= 50 for ideal in results_df['best_ideal_fun'])

        mapped_df = map_test_points_from_results(test_rows, ideal_rows, results_df)
        assert len(mapped_df) == len(test_df)


class TestSummaryPrint:

    def test_summary_print_output(self, caplog):
        import logging
        caplog.set_level(logging.INFO)

        mapped_df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'y': [1, 2, 3, 4],
            'delta_y': [0.1, 0.2, None, None],
            'ideal_function': [1, 1, None, None]
        })

        summary_print(mapped_df)

        assert "Total test points: 4" in caplog.text
        assert "Assigned: 2" in caplog.text
        assert "Unassigned: 2" in caplog.text
