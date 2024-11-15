import numpy as np

import utils.params as params
import utils.evo_data as evo_data
import utils.objectives as objectives
from utils.controller import EvolutionController


def train_head(inputs, prices, device, objective=objectives.cumulative_return, best_model_path=params.BEST_GENOME_PATH, last_model_path=params.LAST_GENOME_PATH, plt_path=params.CR_PLOTS):
    _, num_inputs = inputs.shape
    controller = EvolutionController(num_inputs=num_inputs, device=device, objective=objective, best_model_path=best_model_path, last_model_path=last_model_path, plt_path=plt_path)
    controller.train(inputs, prices)
    return controller
    

if __name__ == '__main__':
    device='cuda'
    print('loading data...')
    in_train, in_test, p_train, p_test = evo_data.get_data(device)
    print('loaded data!')
    train_head(in_train, p_train, device)