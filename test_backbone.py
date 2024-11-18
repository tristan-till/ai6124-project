import torch

import utils.data as data
import utils.test as test
import utils.params as params
import utils.benchmark as benchmark
import utils.plot as plot
import utils.eval as evals

from utils.backbone import get_backbone


def test_backbone(
    num_features,
    test_dataset,
    test_loader,
    device,
    model_path=params.BEST_MODEL_PATH,
    model_loader=get_backbone,
    img_name=params.PLOT_BACKBONE,
):
    model, criterion, _ = model_loader(num_features, device)
    model.load_state_dict(torch.load(f"weights/{model_path}", weights_only=True))
    model.eval()
    predictions, actuals = test.test_backbone(model, criterion, test_loader, device)
    plot.plot_backbone(predictions, actuals, img_name=img_name)
    benchmark.calculate_benchmark(test_dataset)
    evals.evaluate_model(actuals, predictions, device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    dataset, num_features = data.prepare_data()
    
    train_dataset, val_dataset, test_dataset = data.get_datasets(dataset)
    train_loader, val_loader, test_loader = data.get_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    test_backbone(num_features, test_dataset, test_loader, device)
