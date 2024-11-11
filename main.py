import torch

import utils.data as data
import utils.evo_data as evo_data
import utils.params as params
from utils.baseline import get_baseline

from train_baseline import train_baseline
from test_baseline import test_baseline
from train_backbone import train_model
from test_backbone import test_backbone
from finetune_backbone import finetune
from train_head import train_head
from test_head import test_head


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # dataset, num_features = data.prepare_data()
    # train_dataset, val_dataset, test_dataset = data.get_datasets(dataset)
    # train_loader, val_loader, test_loader = data.get_dataloaders(train_dataset, val_dataset, test_dataset)

    # train_baseline(num_features, train_loader, val_loader, device)
    # test_backbone(num_features, test_dataset, test_loader, device, params.BL_BEST_MODEL_PATH, get_baseline, params.PLOT_BASELINE)

    # train_model(num_features, train_loader, val_loader, device)
    # test_backbone(num_features, test_dataset, test_loader, device)

    # finetune(device)
    # finetune(device, params.BL_BEST_MODEL_PATH, get_baseline, params.FT_BL_BEST_MODEL_PATH, params.FT_BL_LAST_MODEL_PATH, params.FT_PLOT_BASELINE)

    # in_train, p_train = evo_data.get_train_data(device, model=params.BEST_MODEL_PATH, target=params.TARGET)
    # train_head(in_train, p_train, device)
    test_head(device)
    test_head(device, params.FT_TARGET, params.FT_BEST_MODEL_PATH, params.FT_BL_BEST_MODEL_PATH, plot_path=params.FT_EVO_PLOT)


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