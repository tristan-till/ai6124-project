import numpy as np

import utils.evo_data as evo_data
from utils.controller import EvolutionController


def train_head(inputs, prices, device):
    _, num_inputs = inputs.shape
    controller = EvolutionController(num_inputs=num_inputs, device=device)
    controller.train(inputs, prices)
    return controller
    

if __name__ == '__main__':
    device='cuda'
    print('loading data...')
    in_train, in_test, p_train, p_test = evo_data.get_data(device)
    print('loaded data!')
    train_head(in_train, p_train, device)