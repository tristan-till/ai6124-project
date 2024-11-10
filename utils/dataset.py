import torch
from torch.utils.data import Dataset

import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # Normalize input data
        self.X = torch.FloatTensor(self._normalize_data(X))
        # Ensure y is 2D
        self.y = torch.FloatTensor(self._normalize_data(y)).reshape(-1, 1)
        
    def _normalize_data(self, data):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # Prevent division by zero
        return (data - mean) / std
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]