import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split

import yfinance as yf

import utils.indicators as indicators
import utils.params as params
import utils.correlation as corr

from utils.dataset import TimeSeriesDataset

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
    return data# , prices

def reshape(X, y, phase=params.PHASE):
    num_samples = len(X) - phase + 1
    X_reshaped = np.array([X[i:i + phase] for i in range(num_samples)])
    y_reshaped = y[phase - 1:]
    return X_reshaped, y_reshaped

def get_target(stock, start_date, end_date=None, phase=params.PHASE):
    target_data = get_data_with_signal(stock, start_date, end_date)
    close_d = target_data[f'{stock}_close_d']
    future_close_d = close_d.shift(-phase)
    future_close_d = future_close_d.dropna()
    x = future_close_d.values
    return x

def get_correlations(stock, stock_data, target_y):
    limit = len(target_y)
    for col in stock_data.columns:
        x = stock_data[col].values[:limit]
        corr_coef, p = corr.get_correlation(x, target_y, stock=stock, name=f"{col}")
        if p < 0.05:
            print(f"{col}", corr_coef, p)

def get_df(supps=params.SUPP):
    X = []
    for stock in supps:
        if len(X) < 1:
            X = get_data_with_signal(stock, params.START_DATE, params.END_DATE)
        else:
            X = pd.merge(X, get_data_with_signal(stock, params.START_DATE, params.END_DATE), on='Date', how='outer')
    return X

def prepare_data(target=params.TARGET, supps=params.SUPP, correlations=False):    
    target_y = get_target(target, params.START_DATE, params.END_DATE, phase=params.PHASE)
    X = get_df(supps)

    if correlations:
        get_correlations(target, X, target_y)

    num_features = len(X.columns)
    X = X.to_numpy()
    y = target_y
    X, y = reshape(X, y, phase=params.PHASE)
    dataset = TimeSeriesDataset(X, y)
    return dataset, num_features

def get_datasets(dataset):
    total_size = len(dataset)
    test_size = int(params.TEST_SIZE * total_size)
    val_size = int(params.VAL_SIZE * total_size)
    train_size = total_size - test_size - val_size

    # Split dataset
    train_indices = range(train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, total_size)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=params.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader