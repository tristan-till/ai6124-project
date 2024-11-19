import torch
import torch.nn.functional as F
from typing import Union

import torch
import numpy as np
import math
import utils.params as params
import ai6124_project.utils.objectives as objectives

def mean_squared_error(y_pred, y_true):
    if y_pred.numel() == 0 or y_true.numel() == 0:
        return torch.tensor(0.0, device=y_pred.device)
    
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    
    return F.mse_loss(y_pred[1:], y_true[1:])

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

def get_evals(y_pred, y_true):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r_squared_value = r_squared(y_true, y_pred)
    hr = hit_ratio(y_true, y_pred)
    return mse, mae, mape, rmse, r_squared_value, hr

def print_evals(title, mse, mae, mape, rmse, r_squared_value, hr):
    print("-" * 100)
    print(
        f"{title}:\n MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, RMSE: {rmse:.4f}, R²: {r_squared_value:.4f}, Hit Ratio: {hr:.2f}%"
    )
    print("-" * 80)


import numpy as np
def evaluate_model(
    y_true: Union[list, torch.Tensor], y_pred: Union[list, torch.Tensor], device
) -> None:
    """
    Evaluate the model using various metrics and print the results.
    Args:
        y_true (Union[list, torch.Tensor]): True values.
        y_pred (Union[list, torch.Tensor]): Predicted values.
    Prints the evaluation metrics.
    """
    r_walk = np.zeros((len(y_true),))
    mean_change = np.ones((len(y_true),)) * np.mean(y_true)
    prev_day = np.insert(np.nan_to_num(np.diff(y_true), 0.0), 0, 0.0)
    prev_day
    y_true, y_pred, r_walk, mean_change, prev_day = (
        torch.from_numpy(y_true),
        torch.from_numpy(y_pred),
        torch.from_numpy(r_walk),
        torch.from_numpy(mean_change),
        torch.from_numpy(prev_day)
    )
    mse, mae, mape, rmse, r_squared_value, hr = get_evals(y_pred, y_true)
    rw_mse, rw_mae, rw_mape, rw_rmse, rw_r_squared_value, rw_hr = get_evals(r_walk, y_true)
    mc_mse, mc_mae, mc_mape, mc_rmse, mc_r_squared_value, mc_hr = get_evals(mean_change, y_true)
    pd_mse, pd_mae, pd_mape, pd_rmse, pd_r_squared_value, pd_hr = get_evals(prev_day, y_true)
    
    print_evals("Current Model", mse, mae, mape, rmse, r_squared_value, hr)
    print_evals("Zero-Change", rw_mse, rw_mae, rw_mape, rw_rmse, rw_r_squared_value, rw_hr)
    print_evals("Mean-Change", mc_mse, mc_mae, mc_mape, mc_rmse, mc_r_squared_value, mc_hr)
    print_evals("Previous-Day-Change", pd_mse, pd_mae, pd_mape, pd_rmse, pd_r_squared_value, pd_hr)

def head_benchmarks(p_test, device):
    ### Buy and Hold ###
    purchasable = math.floor(params.INITIAL_CASH / p_test[0])
    cash = params.INITIAL_CASH - p_test[0] * purchasable * (1 + params.TRANSACTION_FEE)
    stocks = purchasable + params.INITIAL_STOCKS
    b_h_trajectory = torch.Tensor([cash + stocks * p for p in p_test]).to(device)

    ### Risk-Free ###
    annual_risk_free_rate = 0.035
    trading_days_per_year = 252
    daily_risk_free_return = (1 + annual_risk_free_rate) ** (1 / trading_days_per_year) - 1
    initial_price = p_test[0]
    risk_free_trajectory = torch.tensor([
        initial_price * (1 + daily_risk_free_return) ** i for i in range(len(p_test))
    ], device=device)

    ### Cost-Averaging Strategy ###
    avg_trading_days_per_month = 21
    num_months = len(p_test) // avg_trading_days_per_month
    monthly_investment = params.INITIAL_CASH / num_months
    cash = params.INITIAL_CASH
    stocks = params.INITIAL_STOCKS
    ca_trajectory = []
    for day in range(len(p_test)):
        if day % avg_trading_days_per_month == 0:
            purchasable = math.floor(monthly_investment / p_test[day])
            stocks += purchasable
            cash -= purchasable * p_test[day] * (1 + params.TRANSACTION_FEE)
        ca_value = cash + stocks * p_test[day]
        ca_trajectory.append(ca_value)
    ca_trajectory = torch.Tensor(ca_trajectory).to(device)

    b_h_gain = objectives.cumulative_return(b_h_trajectory)
    risk_free_gain = objectives.cumulative_return(risk_free_trajectory)
    ca_gain = objectives.cumulative_return(ca_trajectory)
    # Calculate Sharpe ratios and max drawdowns
    b_h_sharpe = objectives.sharpe_ratio(b_h_trajectory)
    risk_free_sharpe = objectives.sharpe_ratio(risk_free_trajectory)
    ca_sharpe = objectives.sharpe_ratio(ca_trajectory)

    b_h_max_dd = objectives.maximum_drawdown(b_h_trajectory, train=False)
    risk_free_max_dd = objectives.maximum_drawdown(risk_free_trajectory, train=False)
    ca_max_dd = objectives.maximum_drawdown(ca_trajectory, train=False)

    print(f"Buy and Hold: Gain={b_h_gain:.4f}, Sharpe={b_h_sharpe:.4f}, Max Drawdown={b_h_max_dd:.4f}")
    print(f"Risk-Free: Gain={risk_free_gain:.4f}, Sharpe={risk_free_sharpe:.4f}, Max Drawdown={risk_free_max_dd:.4f}")
    print(f"Cost Averaging: Gain={ca_gain:.4f}, Sharpe={ca_sharpe:.4f}, Max Drawdown={ca_max_dd:.4f}")
    

def benchmark_genome():
    custom_gene1 = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    custom_gene2 = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    custom_gene3 = np.array([
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    ])
    custom_gene4 = np.array([
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    ])
    return [custom_gene1, custom_gene2, custom_gene3, custom_gene4]
    