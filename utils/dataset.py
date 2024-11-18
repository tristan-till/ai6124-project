import torch
from torch.utils.data import Dataset

import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        X, self.X_mean, self.x_std = self._normalize_data(X)
        self.X = torch.FloatTensor(X)
        y, self.y_mean, self.y_std = self._normalize_data(y)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        
    def _normalize_data(self, data):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1
        return (data - mean) / std, mean, std
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]