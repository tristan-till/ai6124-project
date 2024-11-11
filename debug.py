import time
import torch

import utils.params as params
import utils.evo_data as evo_data

from utils.controller import EvolutionController

def main():
    device='cuda'
    in_train, p_train = evo_data.get_train_data(device, model=params.BEST_MODEL_PATH, target=params.TARGET)


if __name__ == '__main__':
    main()