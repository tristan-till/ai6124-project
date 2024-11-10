import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import yfinance as yf

import utils.indicators as indicators

seed = 42
random.seed(seed)
np.random.seed(seed)

def get_diff(data):
    data = data.astype(float)
    data = np.diff(data)
    data = np.insert(data, 0, np.nan)
    return data

def get_data_with_signal(stock_code, start_date, end_date=None):
    ticker = yf.Ticker(stock_code)
    df = ticker.history(start=start_date, end=end_date)
    close = df['Close']
    volume = df['Volume']
    
    
    close_d = get_diff(close)
    volume_d = get_diff(volume)
    
    rsi = indicators.get_rsi(close)
    roc = indicators.get_roc(close)
    macd = indicators.get_macd(close)
    macd_diff = get_diff(macd)
    macd_d = indicators.get_macd_diff(close)
    vmacd_s = indicators.get_vmacd_signal(volume)
    
    bb_high = indicators.get_bb_high(close)
    bb_high_d = get_diff(bb_high)
    bb_low = indicators.get_bb_low(close)
    bb_low_d = get_diff(bb_low)
    bb_mid = indicators.get_bb_mid(close)
    bb_mid_d = get_diff(bb_mid)
    
    data = df
    if set(['Open', 'High', 'Low', 'Dividends', 'Stock Splits']).issubset(data.columns):
        data = data.drop(['Open', 'High', 'Low', 'Dividends', 'Stock Splits'], axis=1)
    data['close_d'] = close_d
    data['volume_d'] = volume_d
    data = pd.merge(data, rsi, on='Date', how='outer')
    data = pd.merge(data, roc, on='Date', how='outer')
    data = pd.merge(data, macd, on='Date', how='outer')
    data['macd_diff'] = macd_diff
    data = pd.merge(data, macd_d, on='Date', how='outer')
    data = pd.merge(data, vmacd_s, on='Date', how='outer')
    data = pd.merge(data, bb_high, on='Date', how='outer')
    data = pd.merge(data, bb_low, on='Date', how='outer')
    data = pd.merge(data, bb_mid, on='Date', how='outer')
    data['bb_high_d'] = bb_high_d
    data['bb_low_d'] = bb_low_d
    data['bb_mid_d'] = bb_mid_d
        
    data = data.dropna()
    data = (data - data.min()) / (data.max() - data.min())
    data = data.add_prefix(f'{stock_code}_')
    return data

def reshape(X, y, phase=7):
    num_samples = len(X) - phase + 1
    X_reshaped = np.array([X[i:i + phase] for i in range(num_samples)])
    y_reshaped = y[phase - 1:]
    return X_reshaped, y_reshaped

    
    
def prepare_data(data, target, test_split=0.2, batch_size=32):
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), 
                            torch.tensor(target, dtype=torch.float32).unsqueeze(1))
    
    # Calculate the number of samples for testing and training
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    
    # Randomly split dataset and shuffle training data
    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader