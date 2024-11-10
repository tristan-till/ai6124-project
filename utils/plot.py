import matplotlib.pyplot as plt
import numpy as np

def plot_generation(prices, portfolios, buys, sells, img_path="plot.png"):
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
    ax1.legend(loc='upper left')
    plt.savefig(img_path)
    

def main():
    prices = np.linspace(1, 10, 100) + np.random.uniform(-0.5, 0.5, 100)
    portfolios = [np.linspace(1, 2, 100) + np.random.uniform(-0.5, 0.5, 100)]
    
    plot_generation(prices, portfolios, [[24, 35, 78]], [[66]])

if __name__ == '__main__':
    main()