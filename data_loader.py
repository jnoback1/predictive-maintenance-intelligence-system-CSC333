
"""Load, validate, and prepare predictive maintenance data."""

from pathlib import Path

import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load the AI4I predictive maintenance dataset.

    Args:
        file_path: Location of the CSV dataset.

    Returns:
        A pandas DataFrame containing the machine records.
    """
    return pd.read_csv(file_path)


def validate_data(data: pd.DataFrame) -> None:
    """Validate the machine measurements and failure labels.

    Args:
        data: DataFrame containing machine records.

    Raises:
        ValueError: If the dataset contains invalid or missing data.
    """
    sensor_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    required_columns = ["Type", *sensor_columns, "Machine failure"]

    # Check for an empty dataset.
    if data.empty:
        raise ValueError("The dataset is empty.")

    # Check for missing columns.
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Check for missing values.
    if data[required_columns].isna().any().any():
        raise ValueError("The dataset contains missing values.")

    # Check that sensor measurements are numeric.
    for column in sensor_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"{column} must contain numeric values.")

    # Check for negative sensor measurements.
    if (data[sensor_columns] < 0).any().any():
        raise ValueError("The dataset contains negative sensor measurements.")

    # Check for valid product types.
    if not data["Type"].isin(["L", "M", "H"]).all():
        raise ValueError("The dataset contains invalid product types.")

    # Check for valid failure labels.
    if not data["Machine failure"].isin([0, 1]).all():
        raise ValueError("Machine failure must be either 0 or 1.")


def prepare_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare the input features and machine failure labels.

    Args:
        data: Validated DataFrame containing machine records.

    Returns:
        A tuple containing the input features and failure labels.
    """
    feature_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    # Select the measurements used for prediction.
    X = data[feature_columns].copy()

    # Convert product types into numeric columns.
    X = pd.get_dummies(X, columns=["Type"], dtype=int)

    # Separate the machine failure labels.
    y = data["Machine failure"].copy()

    return X, y
