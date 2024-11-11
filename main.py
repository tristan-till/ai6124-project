import torch

import utils.data as data
import utils.evo_data as evo_data
import utils.params as params

from train_baseline import train_baseline
from test_baseline import test_baseline
from train_backbone import train_model
from test_backbone import test_model
from finetune_backbone import finetune
from train_head import train_head
from test_head import test_head


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    dataset, num_features = data.prepare_data()
    train_dataset, val_dataset, test_dataset = data.get_datasets(dataset)
    train_loader, val_loader, test_loader = data.get_dataloaders(train_dataset, val_dataset, test_dataset)

    train_baseline(num_features, train_loader, val_loader, device)
    test_baseline(num_features, test_dataset, test_loader, device)

    train_model(num_features, train_loader, val_loader, device)
    test_model(num_features, test_dataset, test_loader, device)

    finetune(device)

    in_train, in_test, p_train, p_test = evo_data.get_data(device, model=params.BEST_MODEL_PATH, target=params.TARGET)
    train_head(in_train, p_train, device)
    test_head(in_test, p_test, device=device, plot_path=params.EVO_PLOT)

    _, in_test, _, p_test = evo_data.get_data(device, model=params.FT_BEST_MODEL_PATH, target=params.FT_TARGET)
    test_head(in_test, p_test, device=device, plot_path=params.FT_EVO_PLOT)


if __name__ == '__main__':
    main()

'''
# BACKBONE #
get_backbone_data()
train_baseline_backbone()
train_backbone()
evaluate_backbone()

# HEAD #
get_head_data()
train_baseline_head()
train_head()
evaluate_head()

# TRANSFER LEARNING #
get_backbone_data()
train_backbone()
get_head_data()
# train_head()
evaluate_backbone()
evaluate_head()
'''