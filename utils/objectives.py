import numpy as np

def cumulative_return(history):
    return (history[-1] - history[0]) / history[0]

def sharpe_ratio(history, annual_risk_free_rate=0.035):
    returns = np.diff(history) / history[:-1]
    daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1/252) - 1
    excess_returns = returns - daily_risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_dev_excess_return = np.std(excess_returns)
    sharpe_ratio = mean_excess_return / std_dev_excess_return
    return sharpe_ratio

def maximum_drawdown(history):
    running_max = np.maximum.accumulate(history)
    # Calculate drawdowns
    drawdowns = (running_max - history) / running_max
    max_drawdown = np.max(drawdowns)
    return 1 - max_drawdown

if __name__ == '__main__':
    portfolio_values = [100, 102, 105, 103, 108, 105, 110, 105]
    print(cumulative_return(portfolio_values))
    print(sharpe_ratio(portfolio_values))
    print(maximum_drawdown(portfolio_values))