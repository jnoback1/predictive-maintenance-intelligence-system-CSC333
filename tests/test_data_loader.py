
"""Unit tests for the data loading and preparation module."""

import pandas as pd
import pytest

from data_loader import validate_data, prepare_features


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a small dataset for testing."""
    return pd.DataFrame({
        "Type": ["L", "M"],
        "Air temperature [K]": [298.1, 299.0],
        "Process temperature [K]": [308.6, 309.1],
        "Rotational speed [rpm]": [1551, 1408],
        "Torque [Nm]": [42.8, 46.3],
        "Tool wear [min]": [0, 3],
        "Machine failure": [0, 1],
        "TWF": [0, 0],
        "HDF": [0, 0],
        "PWF": [0, 0],
        "OSF": [0, 0],
        "RNF": [0, 0],
    })


def test_validate_data(sample_data: pd.DataFrame) -> None:
    """Check that valid machine data passes validation."""
    validate_data(sample_data)


def test_negative_tool_wear(sample_data: pd.DataFrame) -> None:
    """Check that negative tool wear is rejected."""
    sample_data.loc[0, "Tool wear [min]"] = -5

    with pytest.raises(ValueError):
        validate_data(sample_data)


def test_prepare_features(sample_data: pd.DataFrame) -> None:
    """Check that features exclude failure labels and failure modes."""
    X, y = prepare_features(sample_data)

    assert "Machine failure" not in X.columns
    assert "TWF" not in X.columns
    assert "Type_L" in X.columns
    assert "Type_M" in X.columns
    assert len(X) == len(y)
