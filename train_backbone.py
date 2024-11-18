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

from utils.eval import evaluate_model


def get_target(stock, target_data, phase=1):
    # target_data = data.get_data_with_signal(stock, start_date, end_date)
    close_d = target_data[f"{stock}_close_d"]
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
    start_date = "2008-01-01"
    end_date = "2024-11-01"
    target = "XOM"
    supp = ["XOM", "OXY", "BP", "^SPX", "COP", "CL=F"]
    # target = "AAPL"
    # supp = ["AAPL", "MSFT", "NVDA", "GOOG", "^SPX"]
    phase = 5

    batch_size = 128
    gru_size = 64
    lstm_size = 16
    attention_size = 32
    num_epochs = 100
    num_layers = 2

    X = []
    for stock in supp:
        if len(X) < 1:
            df = data.get_data_with_signal(stock, start_date, end_date)
            X = df[
                [
                    f"{stock}_close_d",
                    f"{stock}_volume_d",
                    f"{stock}_rsi_week",
                    f"{stock}_rsi_month",
                    f"{stock}_rsi_quarter",
                    f"{stock}_vrsi_week",
                    f"{stock}_vrsi_month",
                    f"{stock}_vrsi_quarter",
                    f"{stock}_roc",
                    f"{stock}_macd",
                    f"{stock}_vmacd_s",
                ]
            ]
        else:
            df = data.get_data_with_signal(stock, start_date, end_date)
            df = df[
                [
                    f"{stock}_close_d",
                    f"{stock}_volume_d",
                    f"{stock}_rsi_week",
                    f"{stock}_rsi_month",
                    f"{stock}_rsi_quarter",
                    f"{stock}_vrsi_week",
                    f"{stock}_vrsi_month",
                    f"{stock}_vrsi_quarter",
                    f"{stock}_roc",
                    f"{stock}_macd",
                    f"{stock}_vmacd_s",
                ]
            ]
            X = pd.merge(X, df, on="Date", how="outer")
    # get_correlations(target, X, target_y)

    num_features = len(X.columns)
    X = X.dropna()
    # target_y = get_target(target, X, phase=phase)
    y = get_target(target, X, phase=phase)
    X = X.to_numpy()
    X_len, _ = X.shape
    len_diff = X_len - len(y)
    X = X[len_diff:, :]
    X, y = data.reshape(X, y, phase=phase)

    # Define dataset
    dataset = TimeSeriesDataset(X, y)

    # Calculate sizes
    total_size = len(dataset)
    test_size = int(0.2 * total_size)
    val_size = int(0.2 * total_size)
    train_size = total_size - test_size - val_size

    # Split dataset
    train_indices = range(train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, total_size)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # benchmark.calculate_benchmark(test_dataset)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )  # Randomize within training subset
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = GRULSTMAttentionModel(
        input_size=num_features,
        gru_size=gru_size,
        lstm_size=lstm_size,
        attention_size=attention_size,
        num_layers=num_layers,
    ).to(device)

    # print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Train the model
    # train.train_model(
    #     model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    # )

    # Test the model
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    # model.load_state_dict(
    #     torch.load("best_model_gru512lstm256attention128.pth", weights_only=True)
    # )
    model.eval()

    df = data.get_data_with_signal(target, start_date, end_date)
    # print(df)
    test = df[f"{target}_Close"][
        # range(train_size + val_size - 1, total_size)
        range(train_size + val_size, total_size)
    ].to_numpy()

    y_true = test[1:]
    r_walk = test[:-1]

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
    print(f"Test Loss: {test_loss:.4f}")

    predictions = np.array(predictions).flatten()[1:]
    actuals = np.array(actuals).flatten()[1:]

    print(predictions.shape, actuals.shape, len(r_walk), len(y_true))

    actuals = actuals * dataset.y_std + dataset.y_mean
    predictions = predictions * dataset.y_std + dataset.y_mean

    y_pred = predictions
    y_true = actuals

    # y_pred, y_true = predictions, actuals

    # actuals = np.array(actuals).flatten()

    diff = np.abs(predictions - actuals)

    # Plot actuals vs. predictions
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="Actual", color="blue")
    plt.plot(predictions, label="Predicted", color="red", linestyle="--")

    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.title("Actual vs. Predicted")
    plt.legend()
    plt.show()

    plt.plot(diff, label="Difference", color="blue")
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.title("Prediction difference")
    plt.show()

    # benchmark.calculate_benchmark(test_dataset)
    evaluate_model(y_true, y_pred, r_walk)

    return predictions, y_true


if __name__ == "__main__":
    main()
