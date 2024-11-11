import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import utils.data as data
import utils.correlation as corr
import utils.train as train
import utils.benchmark as benchmark

from utils.backbone import GRULSTMAttentionModel
from utils.dataset import TimeSeriesDataset

def get_target(stock, start_date, end_date=None, phase=1):
    target_data = data.get_data_with_signal(stock, start_date, end_date)
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

def main():
    start_date = "2020-01-01"
    end_date = "2024-11-01"
    target = "CVX"
    supp = ["CVX", "OXY", "BP", "^SPX", "COP", "CL=F"]	
    # target = "AAPL"
    # supp = ["AAPL", "MSFT", "NVDA", "GOOG", "^SPX"]
    phase=30

    batch_size=16
    gru_size=512
    lstm_size=256
    attention_size=128
    num_epochs=16
    num_layers = 4
    
    target_y = get_target(target, start_date, end_date, phase=phase)
    X = []
    for stock in supp:
        if len(X) < 1:
            X = data.get_data_with_signal(stock, start_date, end_date)
        else:
            X = pd.merge(X, data.get_data_with_signal(stock, start_date, end_date), on='Date', how='outer')
    # get_correlations(target, X, target_y)

    num_features = len(X.columns)
    X = X.dropna()
    X = X.to_numpy()
    y = target_y
    X_len, _ = X.shape
    len_diff = X_len - len(target_y)
    X = X[len_diff:, :]
    X, y = data.reshape(X, y, phase=phase)

    # Define dataset
    dataset = TimeSeriesDataset(X, y)

    # Calculate sizes
    total_size = len(dataset)
    test_size = int(0.2 * total_size)
    val_size = int(0.1 * total_size)
    train_size = total_size - test_size - val_size

    # Split dataset
    train_indices = range(train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, total_size)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    benchmark.calculate_benchmark(test_dataset)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Randomize within training subset
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    
    model = GRULSTMAttentionModel(
        input_size=num_features, 
        gru_size=gru_size, 
        lstm_size=lstm_size, 
        attention_size=attention_size,
        num_layers=num_layers
        ).to(device)

    model.load_state_dict(torch.load('best_model_gru512lstm256attention128.pth', weights_only=True))
    # print(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    
    # Train the model
    train.train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    model = GRULSTMAttentionModel(
        input_size=num_features, 
        gru_size=gru_size, 
        lstm_size=lstm_size, 
        attention_size=attention_size,
        num_layers=num_layers
        ).to(device)
    
    # Test the model
    # model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    model.eval()
    
    test_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            test_loss += criterion(outputs, batch_y).item()
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')

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
    plt.show()



    benchmark.calculate_benchmark(test_dataset)
    
    return predictions, actuals

if __name__ == '__main__':
    main()