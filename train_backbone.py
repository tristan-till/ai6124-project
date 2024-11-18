import torch

import utils.data as data
import utils.train as train
import utils.params as params

from utils.backbone import get_backbone

# def train_model(num_features, train_loader, val_loader, device):    
#     model, criterion, optimizer = get_backbone(num_features, device)
#     train.train_model(model, train_loader, val_loader, criterion, optimizer, device)

def train_backbone(model_loader, num_features, train_loader, val_loader, device, best_model_path=params.BL_BEST_MODEL_PATH, last_model_path=params.BL_LAST_MODEL_PATH):    
    model, criterion, optimizer = model_loader(num_features, device)
    train.train_model(model, train_loader, val_loader, criterion, optimizer, device, best_model_path=best_model_path, last_model_path=last_model_path)

def train_for_folds(num_features, train_dataset, val_dataset, device):
    model, criterion, _ = get_backbone(num_features, device)
    train.train_model_kfold(model, {'input_size': num_features}, train_dataset, val_dataset, criterion, torch.optim.Adam, {'lr': params.LEARNING_RATE, 'weight_decay': params.WEIGHT_DECAY}, device, params.NUM_EPOCHS)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    dataset, num_features = data.prepare_data()
    train_dataset, val_dataset, test_dataset = data.get_datasets(dataset)
    # train_loader, val_loader, test_loader = data.get_dataloaders(train_dataset, val_dataset, test_dataset)
    train_for_folds(num_features, train_dataset, val_dataset, device)