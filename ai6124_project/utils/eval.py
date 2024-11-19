import torch
import torch.nn.functional as F
from typing import Union

import torch
import numpy as np
import math

import ai6124_project.utils.params as params
import ai6124_project.utils.objectives as objectives
import ai6124_project.utils.evo_data as evo_data
from ai6124_project.utils.backbone import get_backbone
import ai6124_project.utils.plot as plot


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
    y_pred_sign = torch.where(y_pred == 0, torch.tensor(1, dtype=y_pred.dtype), torch.sign(y_pred))
    sign_match = torch.sign(y_true.flatten()) == torch.sign(y_pred_sign.flatten())
    hit_rate = torch.mean(sign_match.float()).item() * 100
    return hit_rate


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
    prev_day = np.roll(y_true, shift=1, axis=0)
    y_true, y_pred, r_walk, mean_change, prev_day = (
        torch.from_numpy(y_true),
        torch.from_numpy(y_pred),
        torch.from_numpy(r_walk),
        torch.from_numpy(mean_change),
        torch.from_numpy(prev_day),
    )
    mse, mae, mape, rmse, r_squared_value, hr = get_evals(y_pred, y_true)
    rw_mse, rw_mae, rw_mape, rw_rmse, rw_r_squared_value, rw_hr = get_evals(
        r_walk, y_true
    )
    mc_mse, mc_mae, mc_mape, mc_rmse, mc_r_squared_value, mc_hr = get_evals(
        mean_change, y_true
    )
    pd_mse, pd_mae, pd_mape, pd_rmse, pd_r_squared_value, pd_hr = get_evals(
        prev_day, y_true
    )

    print_evals("Current Model", mse, mae, mape, rmse, r_squared_value, hr)
    print_evals(
        "Zero-Change", rw_mse, rw_mae, rw_mape, rw_rmse, rw_r_squared_value, rw_hr
    )
    print_evals(
        "Mean-Change", mc_mse, mc_mae, mc_mape, mc_rmse, mc_r_squared_value, mc_hr
    )
    print_evals(
        "Previous-Day-Change",
        pd_mse,
        pd_mae,
        pd_mape,
        pd_rmse,
        pd_r_squared_value,
        pd_hr,
    )

from ai6124_project.classes.manager import PortfolioManager
def head_benchmarks(in_test, p_test, model_trajectory, device):
    _, num_inputs = in_test.shape
    bench_manager = PortfolioManager(num_inputs=num_inputs, device=device, rule_operator=lambda x: sum(x) / len(x))
    bench_genome = benchmark_genome()
    bench_manager.fis.set_genome(bench_genome)
    for inp, price in zip(in_test, p_test):
        bench_manager.forward(inp, price)
    bench_trajectory, _, _ = bench_manager.get_history()
    ### Buy and Hold ###
    purchasable = math.floor(params.INITIAL_CASH / p_test[0])
    cash = params.INITIAL_CASH - p_test[0] * purchasable * (1 + params.TRANSACTION_FEE)
    stocks = purchasable + params.INITIAL_STOCKS
    b_h_trajectory = torch.Tensor([cash + stocks * p for p in p_test]).to(device)

    ### Risk-Free ###
    annual_risk_free_rate = 0.035
    trading_days_per_year = 252
    daily_risk_free_return = (1 + annual_risk_free_rate) ** (
        1 / trading_days_per_year
    ) - 1
    initial_price = p_test[0]
    risk_free_trajectory = torch.tensor(
        [initial_price * (1 + daily_risk_free_return) ** i for i in range(len(p_test))],
        device=device,
    )

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

    model_gain = objectives.cumulative_return(model_trajectory)
    bench_gain = objectives.cumulative_return(bench_trajectory)
    b_h_gain = objectives.cumulative_return(b_h_trajectory)
    risk_free_gain = objectives.cumulative_return(risk_free_trajectory)
    ca_gain = objectives.cumulative_return(ca_trajectory)

    model_sharpe = objectives.sharpe_ratio(model_trajectory)
    bench_sharpe = objectives.sharpe_ratio(bench_trajectory)
    b_h_sharpe = objectives.sharpe_ratio(b_h_trajectory)
    risk_free_sharpe = objectives.sharpe_ratio(risk_free_trajectory)
    ca_sharpe = objectives.sharpe_ratio(ca_trajectory)

    model_max_dd = objectives.maximum_drawdown(model_trajectory, train=False)
    bench_max_dd = objectives.maximum_drawdown(bench_trajectory, train=False)
    b_h_max_dd = objectives.maximum_drawdown(b_h_trajectory, train=False)
    risk_free_max_dd = objectives.maximum_drawdown(risk_free_trajectory, train=False)
    ca_max_dd = objectives.maximum_drawdown(ca_trajectory, train=False)

    print(
        f"Model Portfolio: Gain={model_gain:.4f}, Sharpe={model_sharpe:.4f}, Max Drawdown={model_max_dd:.4f}"
    )
    print(
        f"Benchmark Portfolio: Gain={bench_gain:.4f}, Sharpe={bench_sharpe:.4f}, Max Drawdown={bench_max_dd:.4f}"
    )
    print(
        f"Buy and Hold: Gain={b_h_gain:.4f}, Sharpe={b_h_sharpe:.4f}, Max Drawdown={b_h_max_dd:.4f}"
    )
    print(
        f"Risk-Free: Gain={risk_free_gain:.4f}, Sharpe={risk_free_sharpe:.4f}, Max Drawdown={risk_free_max_dd:.4f}"
    )
    print(
        f"Cost Averaging: Gain={ca_gain:.4f}, Sharpe={ca_sharpe:.4f}, Max Drawdown={ca_max_dd:.4f}"
    )

def benchmark_predictions(device, genome_path=params.BEST_CR_GENOME_PATH, model_path=params.BEST_MODEL_PATH, target=params.TARGET, supps=params.SUPP, plot_path="temp.png"):
    base, prices = evo_data.supplementary_data(target)
    model_preds = evo_data.get_predictions(device, model_path=model_path, model_loader=get_backbone, target=target, supps=supps)
    perfect_model_preds = list(np.diff(prices)[-len(model_preds):])
    perfect_model_preds = [float(i)/sum(perfect_model_preds) for i in perfect_model_preds]
    zero_model_preds = [0 for _ in model_preds]
    inverse_model_preds = [1-p for p in model_preds]
    random_signal = list(np.random.uniform(0, 1, len(model_preds)))
            
    _, model_data, _, p_test = evo_data.stack_and_trim(model_preds, prices, base, device)
    _, perfect_model_data, _, _ = evo_data.stack_and_trim(perfect_model_preds, prices, base, device)
    _, zero_model_data, _, _ = evo_data.stack_and_trim(zero_model_preds, prices, base, device)
    _, inverse_model_data, _, _ = evo_data.stack_and_trim(inverse_model_preds, prices, base, device)
    _, random_model_data, _, _ = evo_data.stack_and_trim(random_signal, prices, base, device)
    _, num_inputs = model_data.shape
            
    original_model = PortfolioManager(device, num_inputs)
    zero_model = PortfolioManager(device, num_inputs)
    inverse_model = PortfolioManager(device, num_inputs)
    perfect_model = PortfolioManager(device, num_inputs)
    random_model = PortfolioManager(device, num_inputs)
    
    for manager in [original_model, zero_model, inverse_model, perfect_model, random_model]:
        manager.fis.load_genome(genome_path)
        
    for i, price in enumerate(p_test):
        original_model.forward(model_data[i], price)
        zero_model.forward(perfect_model_data[i], price)
        inverse_model.forward(zero_model_data[i], price)
        perfect_model.forward(inverse_model_data[i], price)
        random_model.forward(random_model_data[i], price)
        
    portfolios, buys, sells = [], [], []
    for manager in [original_model, zero_model, inverse_model, perfect_model, random_model]:
        p, b, s = manager.get_history()
        portfolios.append(p)
        buys.append(b)
        sells.append(s)
    plot.plot_generation(p_test, portfolios, buys, sells, plot_path)
    
    print("---------------------------")
    print(f"### TARGET: {target} ###")
    names = ["Model Predictions", "Perfect Model", "Zero Model", "Inverse Predictions", "Random Signal"]
    for i, portfolio in enumerate(portfolios):
        ca = objectives.cumulative_return(portfolio)
        sharpe = objectives.sharpe_ratio(portfolio)
        max_dd = 1-objectives.maximum_drawdown(portfolio)
        print(f"{names[i]}: Gain={ca:.4f}, Sharpe={sharpe:.4f}, Max Drawdown={max_dd:.4f}")
    print("---------------------------")

def benchmark_genome():
    custom_gene1 = np.array(
        [
            0.1, 0.2, 0.9, # SPX-EMA: Conservative on market weakness, but allows for strong bullish trends
            0.15, 0.25 , 0.5, # SPX-RSI: Very conservative on overbought market conditions
            0.05, 0.15, 0.25, # SPX-MACD: Highly responsive to momentum shifts, use for precise timing
            0.05, 0.15, 0.25, # Target-EMA: Look for moderate move in target asset, assume less volatility
            0.1, 0.2, 0.9, # Target-RSI: Highly responsive to target momentum, strong trend following target
            0.15, 0.25 ,0.5, # Target-MACD: Looks for stronger confirmation in target, use for trend confirmation
            0.15, 0.25, 0.5, # Price-Delta-Prediction: Balanced prediction weighting, use model predictions with caution as they may be inaccurate
            0.05, 0.15, 0.25 # Liquidity: High sensitivity for portfolio liquidity changes, used for risk management
       ]
    )
    custom_gene2 = np.array([
        0.01, 0.25, 0.5, # Low threshold for generating small buy signal, selling small amounts not worth it (transaction fee!), if we sell, sell big
        0.15, 0.35, 0.6, # Slightly higher threshold for holding signal, should only dominate if others are lower
        0.01, 0.25, 0.5]) # Low threshold for generating large buy signal, buying small amounts not worth it (transaction fee!), if we buy, buy big
    custom_gene3 = np.array(
        [
            # SPX-EMA, SPX-RSI, SPX-MACD, Target-EMA, Target-RSI, Target-MACD, Price-Delta-Pred, Liquidity
            [0, 0, 2, 1, 2, 1, 0, 2],  # Bottom formation - strong MACD divergence with good liquidity
            [2, 0, 2, 1, 0, 2, 0, 0],  # Exhaustion rally - strong technicals but poor liquidity
            [2, 1, 2, 2, 1, 1, 1, 1],  # Stable uptrend - strong SPX with confirming target
            [0, 2, 2, 1, 0, 0, 1, 2],  # RSI/MACD divergence with good liquidity - caution signal
            [1, 0, 2, 1, 2, 2, 1, 1],  # Strong momentum divergence between SPX and target
            [0, 0, 1, 0, 1, 1, 0, 1],  # Weak conditions but stabilizing momentum
            [2, 2, 0, 0, 1, 0, 0, 1],  # Overbought with weakening momentum - distribution
            [0, 0, 0, 0, 2, 2, 2, 1],  # Strong target momentum divergence with price prediction - stronger buy signal
            [1, 0, 1, 0, 2, 1, 1, 1],  # Target showing relative strength - accumulation
            [1, 0, 1, 1, 0, 0, 0, 1],  # Mixed signals with weakening target
            [1, 2, 1, 2, 0, 0, 2, 1],  # Overbought with strong price prediction
            [0, 1, 0, 1, 0, 0, 1, 1],  # Weak technicals across board - defensive
            [1, 1, 1, 1, 2, 2, 1, 2],  # Balanced market with strong target momentum
            [1, 1, 1, 1, 1, 1, 1, 1],  # Perfect equilibrium - neutral market
            [0, 1, 1, 0, 2, 1, 0, 1],  # Target strength against weak SPX - watchful
            [0, 1, 1, 0, 1, 1, 1, 1],  # Weak but stabilizing conditions
            [1, 0, 2, 1, 0, 2, 2, 2],  # Strong MACD and prediction with good liquidity - high probability setup
            [0, 0, 0, 0, 1, 1, 1, 1],  # Overall weakness but holding support
            [0, 2, 0, 0, 2, 1, 0, 1],  # Double RSI strength - potential reversal
            [1, 1, 2, 1, 2, 1, 0, 0],  # Strong momentum but poor liquidity setup
        ]
    )
    custom_gene3 = custom_gene3.flatten()
    custom_gene4 = np.array(
        [
            # Sell, Hold, Buy   # Action explanation
            [1, 1, 1],  # Balanced position - early accumulation phase
            [0, 1, 0],  # Hold only - avoid chasing the exhaustion rally
            [0, 2, 0],  # Strong hold - riding the established trend
            [2, 1, 1],  # Reduce with partial hold - protect against weakness
            [2, 0, 1],  # Split position - straddling divergence
            [1, 2, 1],  # Conservative hold - waiting for clarity
            [0, 1, 1],  # Light buying despite distribution - contrarian
            [0, 1, 2],  # [OPTIMIZED] Strong buy on clear target strength divergence
            [0, 0, 1],  # Accumulate slowly - following relative strength
            [0, 1, 1],  # Light buying despite weakness - anticipatory
            [2, 0, 1],  # Reduce but keep small long - mixed signals
            [1, 1, 0],  # Light selling - defensive positioning
            [1, 1, 1],  # Balanced position - healthy market
            [1, 0, 2],  # Shift to bullish in neutral market
            [1, 0, 1],  # Split position - respecting target strength
            [0, 0, 2],  # Strong buy - capitalizing on stabilization
            [0, 1, 2],  # [OPTIMIZED] Confident buy on strong setup with liquidity
            [1, 1, 1],  # Balanced position - awaiting clear direction
            [2, 1, 0],  # Reduce positions - potential double top
            [0, 1, 2],  # Hold with light buying - buy if you can
        ]
    )
    custom_gene4 = custom_gene4.flatten()
    return [custom_gene1, custom_gene2, custom_gene3, custom_gene4]
