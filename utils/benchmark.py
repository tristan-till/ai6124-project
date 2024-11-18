import torch
import torch.nn.functional as F
import numpy as np
import utils.params as params
import math

def calculate_benchmark(test_dataset):
    y_test = torch.stack([sample[1] for sample in test_dataset])

    # Calculate Zero Change Benchmark (predicts zero change for all)
    zero_change_preds = torch.zeros_like(y_test)
    zero_change_loss = F.mse_loss(zero_change_preds, y_test)
    print(f"Zero Change Benchmark MSE Loss: {zero_change_loss.item():.4f}")

    # Calculate Previous Day Change Benchmark
    # Predict the previous day's change for each day in the test set (shift by 1 time step)
    previous_day_change_preds = torch.roll(y_test, shifts=1, dims=0)
    previous_day_change_loss = F.mse_loss(previous_day_change_preds[1:], y_test[1:])  # Skip first element for valid shifting
    print(f"Previous Day Change Benchmark MSE Loss: {previous_day_change_loss.item():.4f}")

    # Calculate Mean Benchmark (predicts the average of y_test for all)
    mean_value = y_test.mean()
    mean_preds = torch.full_like(y_test, mean_value)
    mean_loss = F.mse_loss(mean_preds, y_test)
    print(f"Mean Benchmark MSE Loss: {mean_loss.item():.4f}")

import math
import numpy as np

def head_benchmarks(p_test, device):
    # p_test = p_test.cpu().numpy()
    starting_value = params.INITIAL_CASH + params.INITIAL_STOCKS * p_test[0]
    print(len(p_test), p_test[0], p_test[-1])

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
        if day % avg_trading_days_per_month == 0:  # Buy at the beginning of each month
            purchasable = math.floor(monthly_investment / p_test[day])
            stocks += purchasable
            cash -= purchasable * p_test[day] * (1 + params.TRANSACTION_FEE)
        ca_value = cash + stocks * p_test[day]
        ca_trajectory.append(ca_value)
    ca_trajectory = torch.Tensor(ca_trajectory).to(device)
    print(ca_trajectory[0], ca_trajectory[-1])

    import utils.objectives as objectives
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


# def head_benchmarks(p_test):
#     starting_value = params.INITIAL_CASH + params.INITIAL_STOCKS * p_test[0]
#     print(len(p_test), p_test[0], p_test[-1])
#     ### buy and hold ###
#     purchasable = math.floor(params.INITIAL_CASH / p_test[0])
#     cash = params.INITIAL_CASH - p_test[0] * purchasable * (1+params.TRANSACTION_FEE)
#     stocks = purchasable + params.INITIAL_STOCKS
#     b_h_value = (cash + stocks * p_test[-1] - starting_value) / starting_value
#     ### risk-free ###
#     annual_risk_free_rate=0.035
#     trading_days_per_year = 252
#     daily_risk_free_return = (1 + annual_risk_free_rate) ** (1 / trading_days_per_year) - 1
#     initial_price = p_test[0]
#     portfolio_gain = (initial_price * (1 + daily_risk_free_return) ** len(p_test) - p_test[0]) / p_test[0]
#     print(b_h_value.item(), portfolio_gain.item())
    

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
    