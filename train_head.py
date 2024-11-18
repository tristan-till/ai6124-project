import numpy as np

import utils.params as params
import utils.evo_data as evo_data
import utils.objectives as objectives
from utils.controller import EvolutionController


def train_head(inputs, prices, device, objective=objectives.cumulative_return, best_model_path=params.BEST_GENOME_PATH, last_model_path=params.LAST_GENOME_PATH, plt_path=params.CR_PLOTS):
    _, num_inputs = inputs.shape
    controller = EvolutionController(num_inputs=num_inputs, device=device, objective=objective, best_model_path=best_model_path, last_model_path=last_model_path, plt_path=plt_path)
    controller.train(inputs, prices)
    

if __name__ == '__main__':
    device='cuda'
    in_train, p_train = evo_data.get_train_data(device, model=params.BEST_MODEL_PATH, target=params.TARGET)
    train_head(in_train, p_train, device, objective=objectives.cumulative_return, best_model_path=params.BEST_CR_GENOME_PATH, last_model_path=params.LAST_CR_GENOME_PATH, plt_path=params.CR_PLOTS)
    train_head(in_train, p_train, device, objective=objectives.sharpe_ratio, best_model_path=params.BEST_SR_GENOME_PATH, last_model_path=params.LAST_SR_GENOME_PATH, plt_path=params.SR_PLOTS)
    train_head(in_train, p_train, device, objective=objectives.maximum_drawdown, best_model_path=params.BEST_MD_GENOME_PATH, last_model_path=params.LAST_MD_GENOME_PATH, plt_path=params.MD_PLOTS)
