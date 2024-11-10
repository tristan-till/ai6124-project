import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def get_correlation(x, y, stock, name='corr'):
  colors = plt.colormaps['Blues'](np.linspace(0, 1, len(x)))

  plt.grid(True)
  plt.scatter(x, y, color=colors)
  coeffs = np.polyfit(x, y, deg=1)
  corr_line = np.poly1d(coeffs)
  x_corr = np.linspace(x.min(), x.max(), 100)
  corr_coef, p_value = pearsonr(x, y)
  plt.plot(x_corr, corr_line(x_corr), color='red', linewidth=2, label='Correlation Line (c = {:.2f}, p = {:.2f})'.format(coeffs[0], p_value))
  plt.legend()
  path = f"plots/correlations/{stock}"
  if not os._exists(f"{path}/"):
    os.makedirs(f"{path}/", exist_ok=True)
  plt.savefig(f'{path}/{name}.png', dpi=150)
  plt.close()
  
  return corr_coef, p_value