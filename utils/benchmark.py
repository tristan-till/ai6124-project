import torch
import torch.nn.functional as F

def calculate_benchmark(test_dataset):
    y_test = torch.stack([sample[1] for sample in test_dataset])

    # Calculate Zero Change Benchmark (predicts zero change for all)
    zero_change_preds = torch.zeros_like(y_test)
    zero_change_loss = F.mse_loss(zero_change_preds, y_test)
    print(f"Zero Change Benchmark MSE Loss: {zero_change_loss.item():.4f}")

    # Calculate Previous Day Change Benchmark
    # Predict the previous day's change for each day in the test set (shift by 1 time step)
    previous_day_change_preds = torch.roll(y_test, shifts=1, dims=0)
    previous_day_change_loss = F.mse_loss(previous_day_change_preds[1:], y_test[1:])  # Skip first element for valid shifting
    print(f"Previous Day Change Benchmark MSE Loss: {previous_day_change_loss.item():.4f}")

    # Calculate Mean Benchmark (predicts the average of y_test for all)
    mean_value = y_test.mean()
    mean_preds = torch.full_like(y_test, mean_value)
    mean_loss = F.mse_loss(mean_preds, y_test)
    print(f"Mean Benchmark MSE Loss: {mean_loss.item():.4f}")

