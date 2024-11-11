import torch

import utils.data as data
import utils.test as test
import utils.params as params
import utils.benchmark as benchmark
import utils.plot as plot
import utils.train as train

from utils.backbone import get_backbone

def finetune(device, model_path=params.BEST_MODEL_PATH, model_loader=get_backbone, 
             best_model_path=params.FT_BEST_MODEL_PATH, last_model_path=params.FT_LAST_MODEL_PATH,
             plot_name=params.FT_PLOT_BACKBONE):
    dataset, num_features = data.prepare_data(params.FT_TARGET, params.FT_SUPP)
    train_dataset, val_dataset, test_dataset = data.get_datasets(dataset)
    train_loader, val_loader, test_loader = data.get_dataloaders(train_dataset, val_dataset, test_dataset)

    model, criterion, optimizer = model_loader(num_features, device)
    model.load_state_dict(torch.load(f"weights/{model_path}", weights_only=True))
    train.train_model(model, train_loader, val_loader, criterion, optimizer, device, best_model_path=best_model_path, last_model_path=last_model_path)
    model.eval()
    predictions, actuals = test.test_backbone(model, criterion, test_loader, device)
    plot.plot_backbone(predictions, actuals, plot_name)
    benchmark.calculate_benchmark(test_dataset)
    return predictions, actuals

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    finetune(device)