import torch

def cumulative_return(history):
    return (history[-1] - history[0]) / history[0]

def sharpe_ratio(history, annual_risk_free_rate=0.035):
    returns = (history[1:] - history[:-1]) / history[:-1]
    daily_risk_free_rate = torch.tensor((1 + annual_risk_free_rate) ** (1/252) - 1, device=history.device)
    excess_returns = returns - daily_risk_free_rate
    valid_mask = torch.isfinite(excess_returns)
    valid_excess_returns = excess_returns[valid_mask]
    if torch.allclose(valid_excess_returns, torch.tensor(0.0, device=history.device), atol=1e-6, rtol=0):
        return torch.tensor(0.0, device=history.device)

    # Check for empty or insufficient valid data
    if valid_excess_returns.numel() == 0:
        print("Warning: No valid data available for Sharpe ratio calculation.")
        return torch.tensor(float('nan'), device=history.device)
    mean_excess_return = torch.mean(valid_excess_returns)
    std_dev_excess_return = torch.std(valid_excess_returns)
    sharpe_ratio = mean_excess_return / std_dev_excess_return
    return sharpe_ratio

def maximum_drawdown(history, train=True):
    running_max = torch.cummax(history, dim=0)[0]
    drawdowns = (running_max - history) / running_max
    max_drawdown = torch.max(drawdowns)
    return 1 - max_drawdown if train else max_drawdown

if __name__ == '__main__':
    portfolio_values = [100, 102, 105, 103, 108, 105, 110, 105]
    print(cumulative_return(portfolio_values))
    print(sharpe_ratio(portfolio_values))
    print(maximum_drawdown(portfolio_values))