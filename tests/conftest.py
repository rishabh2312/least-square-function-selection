import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile


class MockRow:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def sample_x_values():
    return np.arange(-20.0, 20.1, 0.1)


@pytest.fixture
def sample_training_df(sample_x_values):
    x = sample_x_values
    return pd.DataFrame({
        'x': x,
        'y1': x ** 2,
        'y2': np.sin(x),
        'y3': x,
        'y4': np.exp(-x / 10)
    })


@pytest.fixture
def sample_ideal_df(sample_x_values):
    x = sample_x_values
    data = {'x': x}
    data['y1'] = x ** 2
    data['y2'] = np.sin(x)
    data['y3'] = x
    data['y4'] = np.exp(-x / 10)
    for i in range(5, 51):
        data[f'y{i}'] = np.random.randn(len(x)) * 100
    return pd.DataFrame(data)


@pytest.fixture
def sample_test_df():
    return pd.DataFrame({
        'x': [-10.0, -5.0, 0.0, 5.0, 10.0],
        'y': [100.0, 25.0, 0.0, 25.0, 100.0]
    })


@pytest.fixture
def mock_training_rows(sample_training_df):
    return [MockRow(**row.to_dict()) for _, row in sample_training_df.iterrows()]


@pytest.fixture
def mock_ideal_rows(sample_ideal_df):
    return [MockRow(**row.to_dict()) for _, row in sample_ideal_df.iterrows()]


@pytest.fixture
def mock_test_rows(sample_test_df):
    return [MockRow(**row.to_dict()) for _, row in sample_test_df.iterrows()]


@pytest.fixture
def temp_data_dir(sample_training_df, sample_ideal_df, sample_test_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        sample_training_df.to_csv(data_dir / "train.csv", index=False)
        sample_ideal_df.to_csv(data_dir / "ideal.csv", index=False)
        sample_test_df.to_csv(data_dir / "test.csv", index=False)
        yield Path(tmpdir)
