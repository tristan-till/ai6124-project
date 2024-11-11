import time
import torch

import utils.params as params
import utils.evo_data as evo_data

from utils.controller import EvolutionController

def main():
    ins = 50
    device='cuda'
    in_train, in_test, p_train, p_test = evo_data.get_data(device, model=params.BEST_MODEL_PATH, target=params.TARGET)  
    _, num_inputs = in_train.shape
    controller = EvolutionController(device, num_inputs)

    start_time = time.time()
    # Process the selected sequence
    controller.train(in_train, p_train)

    end_time = time.time()
    time_diff = end_time - start_time
    print(f"Training time: {time_diff:.2f} seconds")


if __name__ == '__main__':
    main()