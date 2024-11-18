import utils.data as data
import utils.params as params
from utils.backbone import get_backbone

import yfinance as yf
import torch
import pandas as pd
import numpy as np


def get_prices(stock=params.TARGET):
    ticker = yf.Ticker(stock)
    df = ticker.history(start=params.START_DATE, end=params.END_DATE)
    df = df.dropna()
    return df['Close']

def get_predictions(device, model_path=params.BEST_MODEL_PATH, model_loader=get_backbone):
    dataset, num_features = data.prepare_data()
    train_dataset, val_dataset, test_dataset = data.get_datasets(dataset)
    train_loader, val_loader, test_loader = data.get_dataloaders(train_dataset, val_dataset, test_dataset)
    model, _, _ = model_loader(num_features, 'cuda')
    model.load_state_dict(torch.load(f"weights/{model_path}", weights_only=True))

    preds = []
    
    with torch.no_grad():
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds.extend(outputs.cpu().numpy())
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds.extend(outputs.cpu().numpy())
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds.extend(outputs.cpu().numpy())
    
    return preds

def load_data(stock):
    ticker = yf.Ticker(stock)
    df = ticker.history(start=params.START_DATE, end=params.END_DATE)
    df = df.dropna()
    # df = df.drop(['Open', 'High', 'Low', 'Dividends', 'Stock Splits'], axis=1)
    df = df.drop(['Volume', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits'], axis=1)
    df = (df - df.min()) / (df.max() - df.min())
    return df

def other_data(stocks):
    X = []
    for stock in stocks:
        if len(X) < 1:
            X = load_data(stock)
        else:
            X = pd.merge(X, load_data(stock), on='Date', how='outer')
    return X

def supplementary_data(target):
    prices = get_prices(target)
    other = other_data(["^SPX", target])
    return other, prices

def load(model, target, device, model_loader):
    preds = get_predictions(device, model, model_loader)
    other, prices = supplementary_data(target)
    train_in, test_in, train_p, test_p = stack_and_trim(preds, prices, other, device)
    return train_in, test_in, train_p, test_p

def stack_and_trim(preds, prices, other, device):
    base = other.copy()
    diff = len(prices) - len(preds)
    prices = np.array(prices[diff:])
    base = base[diff:]
    base['preds'] = np.array(preds).flatten()
    inputs = base.to_numpy(dtype=np.float32)
    in_train, in_test, p_train, p_test = split_data(inputs, prices)
    return torch.Tensor(in_train).to(device), torch.Tensor(in_test).to(device), torch.Tensor(p_train).to(device), torch.Tensor(p_test).to(device)

def stack_on_base(base, arr, device):
    stack = base.copy()
    stack['preds'] = arr.flatten()
    return torch.Tensor(stack.to_numpy()).to(device)

def get_trivial_predictions(base, prices, device):
    np_prices = prices.cpu().numpy()
    num_samples = len(np_prices)
    test_base = base.copy()[-num_samples:]
    zero_change = np.zeros((num_samples,))
    random = np.random.uniform(-1, 1, (num_samples,))
    perfect_preds = np.append(np.diff(np_prices), 0.0)
    trivial_preds = [0]
    trivial_preds.extend(perfect_preds[:-1])
    perfect_preds = np.nan_to_num(perfect_preds, nan=0.0, posinf=None, neginf=None)
    trivial_preds = np.nan_to_num(trivial_preds, nan=0.0, posinf=None, neginf=None)
    zero_change_in = stack_on_base(test_base, zero_change, device)
    random_in = stack_on_base(test_base, random, device)
    perf_pred_in = stack_on_base(test_base, perfect_preds, device)
    triv_pred_in = stack_on_base(test_base, trivial_preds, device)
    return zero_change_in, random_in, perf_pred_in, triv_pred_in

def get_train_data(device, model=params.BEST_MODEL_PATH, target=params.TARGET, model_loader=get_backbone):
    in_train, _, p_train, _ = load(model, target, device, model_loader)
    return in_train, p_train

def get_test_data(device, model=params.BEST_MODEL_PATH, target=params.TARGET, model_loader=get_backbone):
    _, in_test, _, p_test = load(model, target, device, model_loader)
    return in_test, p_test

def split_data(inputs, prices):
    split_index = int((1 - params.TEST_SIZE) * len(inputs))
    in_train, in_test = inputs[:split_index], inputs[split_index:]
    p_train, p_test = prices[:split_index], prices[split_index:]
    return in_train, in_test, p_train, p_test