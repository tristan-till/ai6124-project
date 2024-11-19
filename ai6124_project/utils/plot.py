import os
from threading import Lock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import ai6124_project.utils.params as params

plot_lock = Lock()

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
    # Convert tensors to numpy arrays outside the lock
    prices = prices.cpu().numpy() if isinstance(prices, torch.Tensor) else prices
    portfolios = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in portfolios]
    buys = [b.cpu().numpy() if isinstance(b, torch.Tensor) else b for b in buys]
    sells = [s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in sells]
    
    with plot_lock:
        try:
            # Create a new figure with a specific ID
            fig = plt.figure()
            
            x_len = len(prices)
            x = np.linspace(0, x_len-1, x_len)
            
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            
            # Plot data
            ax2.plot(x, prices, zorder=1, color='black', linewidth=1, label='Prices')
            
            for i, portfolio in enumerate(portfolios):
                ax1.plot(x, portfolio, zorder=2, linewidth=0.5, label=f'Portfolio {i}')
                buy = buys[i]
                for b in buy:
                    ax1.scatter(b, portfolio[b], color='g', marker='^', s=15, zorder=2)
                sell = sells[i]
                for s in sell:
                    ax1.scatter(s, portfolio[s], color='r', marker='v', s=15, zorder=2)
            
            # Ensure the output directory exists
            img_dest = os.path.join(params.PLOT_PATH, img_name)
            dest_path = os.path.dirname(img_dest)
            if not os._exists(dest_path):
                os.makedirs(dest_path, exist_ok=True)
            
            # Save the figure
            fig.savefig(img_dest, dpi=150)
            
            # Explicitly close the figure
            plt.close(fig)
            
        except Exception as e:
            print(f"Error in plot_generation: {str(e)}")
        finally:
            # Ensure we always try to close any remaining figures
            try:
                plt.close('all')
            except Exception as e:
                print(f"Error closing figures: {str(e)}")
    

def main():
    prices = np.linspace(1, 10, 100) + np.random.uniform(-0.5, 0.5, 100)
    portfolios = [np.linspace(1, 2, 100) + np.random.uniform(-0.5, 0.5, 100)]
    
    plot_generation(prices, portfolios, [[24, 35, 78]], [[66]])

if __name__ == '__main__':
    main()