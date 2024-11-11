import torch

import utils.data as data
import utils.test as test
import utils.params as params
import utils.benchmark as benchmark
import utils.plot as plot

from utils.baseline import get_baseline

def test_baseline(num_features, test_dataset, test_loader, device, model_path=params.BL_BEST_MODEL_PATH):
    model, criterion, _ = get_baseline(num_features, device)
    model.load_state_dict(torch.load(f"weights/{model_path}", weights_only=True))
    model.eval()
    predictions, actuals = test.test_backbone(model, criterion, test_loader, device)
    plot.plot_backbone(predictions, actuals, "baseline.png")
    benchmark.calculate_benchmark(test_dataset)
    return predictions, actuals

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    dataset, num_features = data.prepare_data()
    train_dataset, val_dataset, test_dataset = data.get_datasets(dataset)
    train_loader, val_loader, test_loader = data.get_dataloaders(train_dataset, val_dataset, test_dataset)
    main()