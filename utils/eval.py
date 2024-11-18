import torch
from typing import Union


def hit_ratio(
    y_true: Union[list, torch.Tensor], y_pred: Union[list, torch.Tensor]
) -> float:
    """
    Calculate the hit ratio between true and predicted values.

    Args:
        y_true (Union[list, torch.Tensor]): True values.
        y_pred (Union[list, torch.Tensor]): Predicted values.

    Returns:
        float: Hit ratio as a percentage.
    """
    # Calculate hit ratio
    correct_predictions = (
        torch.sign(y_true[1:] - y_true[:-1]) == torch.sign(y_pred[1:] - y_pred[:-1])
    ).float()

    return torch.mean(correct_predictions).item() * 100


def mean_absolute_error(
    y_true: Union[list, torch.Tensor], y_pred: Union[list, torch.Tensor]
) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Args:
        y_true (Union[list, torch.Tensor]): True values.
        y_pred (Union[list, torch.Tensor]): Predicted values.

    Returns:
        float: MAE value.
    """
    return torch.mean(torch.abs(y_true - y_pred)).item()


def mean_absolute_percentage_error(
    y_true: Union[list, torch.Tensor], y_pred: Union[list, torch.Tensor]
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true (Union[list, torch.Tensor]): True values.
        y_pred (Union[list, torch.Tensor]): Predicted values.

    Returns:
        float: MAPE value as a percentage.
    """
    return torch.mean(torch.abs((y_true - y_pred) / y_true)).item() * 100


def root_mean_squared_error(
    y_true: Union[list, torch.Tensor], y_pred: Union[list, torch.Tensor]
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    Args:
        y_true (Union[list, torch.Tensor]): True values.
        y_pred (Union[list, torch.Tensor]): Predicted values.

    Returns:
        float: RMSE value.
    """
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def r_squared(
    y_true: Union[list, torch.Tensor], y_pred: Union[list, torch.Tensor]
) -> float:
    """
    Calculate R-squared value.

    Args:
        y_true (Union[list, torch.Tensor]): True values.
        y_pred (Union[list, torch.Tensor]): Predicted values.

    Returns:
        float: R-squared value.
    """
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()

    return 1 - (ss_res / ss_tot)


def evaluate_model(
    y_true: Union[list, torch.Tensor], y_pred: Union[list, torch.Tensor], r_walk
) -> None:
    """
    Evaluate the model using various metrics and print the results.

    Args:
        y_true (Union[list, torch.Tensor]): True values.
        y_pred (Union[list, torch.Tensor]): Predicted values.

    Prints the evaluation metrics.
    """

    y_true, y_pred, r_walk = (
        torch.from_numpy(y_true),
        torch.from_numpy(y_pred),
        torch.from_numpy(r_walk),
    )

    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    r_squared_value = r_squared(y_true, y_pred)

    hr = hit_ratio(y_true, y_pred)

    rw_mae = mean_absolute_error(y_true, r_walk)
    rw_mape = mean_absolute_percentage_error(y_true, r_walk)
    rw_rmse = root_mean_squared_error(y_true, r_walk)

    rw_r_squared_value = r_squared(y_true, r_walk)

    rw_hr = hit_ratio(y_true, r_walk)

    print("-" * 100)
    print(
        f"Current Model:\n MAE: {mae:.4f}, MAPE: {mape:.2f}%, RMSE: {rmse:.4f}, R²: {r_squared_value:.4f}, Hit Ratio: {hr:.2f}%"
    )

    print("-" * 80)

    print(
        f"Random Walk:\n MAE: {rw_mae:.4f}, MAPE: {rw_mape:.2f}%, RMSE: {rw_rmse:.4f}, R²: {rw_r_squared_value:.4f}, Hit Ratio: {rw_hr:.2f}%"
    )
