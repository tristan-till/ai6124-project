import torch

import ai6124_project.utils.data as data
import ai6124_project.utils.test as test
import ai6124_project.utils.params as params
import ai6124_project.utils.plot as plot
import ai6124_project.utils.train as train
import ai6124_project.utils.eval as evals

from ai6124_project.utils.backbone import get_backbone, get_baseline

def finetune(device, num_features, train_loader, val_loader, test_loader, model_path=params.BEST_MODEL_PATH, model_loader=get_backbone, num_epochs=params.FT_NUM_EPOCHS,
             best_model_path=params.FT_BEST_MODEL_PATH, last_model_path=params.FT_LAST_MODEL_PATH,
             plot_name=params.FT_PLOT_BACKBONE):
    model, criterion, optimizer = model_loader(num_features, device)
    model.load_state_dict(torch.load(f"weights/{model_path}", weights_only=True, map_location=device))
    train.train_model(model, train_loader, val_loader, criterion, optimizer, device, best_model_path=best_model_path, last_model_path=last_model_path, num_epochs=num_epochs)
    eval_model, criterion, _ = model_loader(num_features, device)
    model.load_state_dict(torch.load(f"weights/{best_model_path}", weights_only=True, map_location=device))
    eval_model.eval()
    predictions, actuals = test.test_backbone(eval_model, criterion, test_loader, device)
    plot.plot_backbone(predictions, actuals, plot_name)
    evals.evaluate_model(actuals, predictions, device)
    return predictions, actuals

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    dataset, num_features = data.prepare_data(params.FT_TARGET, params.FT_SUPP)
    train_dataset, val_dataset, test_dataset = data.get_datasets(dataset)
    train_loader, val_loader, test_loader = data.get_dataloaders(train_dataset, val_dataset, test_dataset)
    finetune(device)
    finetune(device, params.BL_BEST_MODEL_PATH, get_baseline, params.FT_NUM_EPOCHS, params.FT_BL_BEST_MODEL_PATH, params.FT_BL_LAST_MODEL_PATH, params.FT_PLOT_BASELINE)