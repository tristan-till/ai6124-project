import torch

import ai6124_project.utils.data as data
import ai6124_project.utils.test as test
import ai6124_project.utils.params as params
import ai6124_project.utils.plot as plot
import ai6124_project.utils.eval as evals

from ai6124_project.utils.backbone import get_backbone, get_baseline


def test_backbone(
    num_features,
    test_loader,
    device,
    model_path=params.BEST_MODEL_PATH,
    model_loader=get_backbone,
    img_name=params.PLOT_BACKBONE,
):
    model, criterion, _ = model_loader(num_features, device)
    model.load_state_dict(torch.load(f"weights/{model_path}", weights_only=True, map_location=device))
    model.eval()
    predictions, actuals = test.test_backbone(model, criterion, test_loader, device)
    plot.plot_backbone(predictions, actuals, img_name=img_name)
    evals.evaluate_model(actuals, predictions, device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    import ai6124_project.utils.helpers as helpers
    helpers.set_deterministic()
    dataset, num_features = data.prepare_data()
    
    train_dataset, val_dataset, test_dataset = data.get_datasets(dataset)
    train_loader, val_loader, test_loader = data.get_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    test_backbone(num_features, test_loader, device)
    test_backbone(num_features, test_loader, device, params.BL_BEST_MODEL_PATH, get_baseline, params.PLOT_BASELINE)
