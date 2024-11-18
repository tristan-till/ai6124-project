import torch

import utils.data as data
import utils.train as train
import utils.params as params

from utils.baseline import get_baseline

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    dataset, num_features = data.prepare_data()
    train_dataset, val_dataset, test_dataset = data.get_datasets(dataset)
    train_loader, val_loader, test_loader = data.get_dataloaders(train_dataset, val_dataset, test_dataset)
    train_model(num_features, train_loader, val_loader, device)