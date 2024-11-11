import matplotlib.pyplot as plt
import numpy as np
import torch

import utils.params as params

def plot_backbone(predictions, actuals, img_name="backbone.png"):
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # Plot actuals vs. predictions
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label='Actual', color='blue')
    plt.plot(predictions, label='Predicted', color='red', linestyle='--')

    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Actual vs. Predicted')
    plt.legend()
    plt.savefig(f"{params.PLOT_PATH}/{img_name}")

def plot_generation(prices, portfolios, buys, sells, img_name="head.png"):
    prices = prices.cpu().numpy() if isinstance(prices, torch.Tensor) else prices
    portfolios = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in portfolios]
    buys = [b.cpu().numpy() if isinstance(b, torch.Tensor) else b for b in buys]
    sells = [s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in sells]

    x_len = len(prices)
    x = np.linspace(0, x_len-1, x_len)
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.plot(x, prices, zorder=1, color='black', linewidth=1, label='Prices')

    for i, portfolio in enumerate(portfolios):
        ax1.plot(x, portfolio, zorder=2, linewidth=0.5, label=f'Portfolio {i}')
        buy = buys[i]
        for b in buy:
            ax1.scatter(b, portfolio[b], color='g', marker='^', s=15, zorder=2)
        sell = sells[i]
        for s in sell:
            ax1.scatter(s, portfolio[s], color='r', marker='v', s=15, zorder=2)

    plt.savefig(f"{params.PLOT_PATH}/{img_name}", dpi=150)
    

def main():
    prices = np.linspace(1, 10, 100) + np.random.uniform(-0.5, 0.5, 100)
    portfolios = [np.linspace(1, 2, 100) + np.random.uniform(-0.5, 0.5, 100)]
    
    plot_generation(prices, portfolios, [[24, 35, 78]], [[66]])

if __name__ == '__main__':
    main()