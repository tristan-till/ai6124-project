import torch

import utils.data as data
import utils.train as train

from utils.backbone import get_backbone

def train_model(num_features, train_loader, val_loader, device):    
    model, criterion, optimizer = get_backbone(num_features, device)
    train.train_model(model, train_loader, val_loader, criterion, optimizer, device)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    dataset, num_features = data.prepare_data()
    train_dataset, val_dataset, test_dataset = data.get_datasets(dataset)
    train_loader, val_loader, test_loader = data.get_dataloaders(train_dataset, val_dataset, test_dataset)
    train_model(num_features, train_loader, val_loader, device)