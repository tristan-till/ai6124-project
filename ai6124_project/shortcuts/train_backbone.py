import torch

import ai6124_project.utils.data as data
import ai6124_project.utils.train as train
import ai6124_project.utils.params as params

from ai6124_project.utils.backbone import get_backbone
from ai6124_project.utils.backbone import get_baseline

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
    train_loader, val_loader, test_loader = data.get_dataloaders(train_dataset, val_dataset, test_dataset)
    train_backbone(get_backbone, num_features, train_loader, test_loader, device, params.BEST_MODEL_PATH, params.LAST_MODEL_PATH)
    train_backbone(get_baseline, num_features, train_loader, test_loader, device, params.BL_BEST_MODEL_PATH, params.BL_LAST_MODEL_PATH)
    # train_for_folds(num_features, train_dataset, val_dataset, device)