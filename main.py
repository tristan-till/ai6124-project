import numpy as np
import pandas as pd

import torch

import utils.data as data

from utils.backbone import GRULSTMAttentionModel

# from utils.controller import EvolutionController


def create_convolved_features_array(X, p):
    """
    Create convolved features from a DataFrame X for a given number of previous days p.

    Parameters:
    X (pd.DataFrame): Input DataFrame with features.
    p (int): Number of previous days to include as features.

    Returns:
    np.ndarray: A 3D NumPy array with shape (len(X), p, n) where n is the number of features.
    """

    # Ensure that p is a positive integer
    if not isinstance(p, int) or p <= 0:
        raise ValueError("Parameter 'p' must be a positive integer.")

    # Get the number of rows and columns in the DataFrame
    num_rows, num_features = X.shape

    # Initialize an empty array to hold the convolved features
    convolved_array = np.full((num_rows, p, num_features), np.nan)

    # Create an index array for slicing
    indices = np.arange(p).reshape(1, -1) + np.arange(num_rows - p + 1).reshape(-1, 1)

    # Fill the convolved array using advanced indexing
    convolved_array[p - 1 :, :, :] = X.values[indices]

    return convolved_array


def split_data(X, y, d, v):
    """
    Split data into training, validation, and test sets.

    Parameters:
    X (pd.DataFrame or np.ndarray): Input features.
    y (pd.Series or np.ndarray): Target variable.
    d (int): Number of days to predict (test set size).
    v (int): Validation set size.

    Returns:
    tuple: A tuple containing (X_train, y_train), (X_val, y_val), (X_test, y_test).
    """

    # Ensure that d and v are positive integers
    if not (isinstance(d, int) and d > 0) or not (isinstance(v, int) and v > 0):
        raise ValueError("Parameters 'd' and 'v' must be positive integers.")

    # Convert to numpy arrays if they are in DataFrame/Series format
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # Total number of samples
    total_samples = len(y)

    # Calculate the indices for splitting
    test_start_index = total_samples - d
    val_start_index = test_start_index - v

    # Ensure there are enough samples for validation and testing
    if val_start_index < 0:
        raise ValueError(
            "Not enough data points for the specified validation and test sizes."
        )

    # Split the data
    X_train = X[:val_start_index]
    y_train = y[:val_start_index]

    X_val = X[val_start_index:test_start_index]
    y_val = y[val_start_index:test_start_index]

    X_test = X[test_start_index:]
    y_test = y[test_start_index:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# def get_target(stock, start_date, end_date=None, phase=1):
#     target_data = data.get_data_with_signal(stock, start_date, end_date)
#     close_d = target_data[f'{stock}_close_d']
#     future_close_d = close_d.shift(-phase)
#     future_close_d = future_close_d.dropna()
#     x = future_close_d.values
#     return x


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    start_date = "2008-10-01"
    end_date = "2024-11-01"
    target = "XOM"
    phase = 5
    future = 1

    df = data.get_data_with_signal(target, start_date, end_date)
    # prices = np.array(df['XOM_Close'].values[-100:])

    df = df[
        [
            "XOM_close_d",
            "XOM_volume_d",
            "XOM_rsi_week",
            "XOM_rsi_month",
            "XOM_rsi_quarter",
            "XOM_vrsi_week",
            "XOM_vrsi_month",
            "XOM_vrsi_quarter",
            "XOM_roc",
            "XOM_macd",
            "XOM_vmacd_s",
        ]
    ]

    X, y = df.iloc[:, :-future], df["XOM_close_d"][future:]
    X = create_convolved_features_array(X, phase)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y, 300, 100)

    print(X.shape, y.shape)

    # target_y = get_target(target, start_date, end_date, phase=phase)[-100:]

    # X = np.stack([f1, f2, f3, f4, target_y], axis=1)

    num_features = 96

    gru_size = 512
    lstm_size = 256
    attention_size = 128
    num_layers = 4
    predictor = GRULSTMAttentionModel(
        input_size=num_features,
        gru_size=gru_size,
        lstm_size=lstm_size,
        attention_size=attention_size,
        num_layers=num_layers,
    ).to(device)
    predictor.train(X, y)
    # controller = EvolutionController(10, 10)
    # controller.train(X, prices)


if __name__ == "__main__":
    main()
