import time
import torch
import numpy as np

import utils.params as params
import utils.evo_data as evo_data
import utils.benchmark as benchmark
import utils.enums as enums

from utils.aggregation import AggregationLayer
from utils.manager import PortfolioManager

def main():
    device='cuda'
    # in_test, p_test = evo_data.get_test_data(device)
    # benchmark.head_benchmarks(p_test, device)
    in_test, p_test = evo_data.get_test_data(device, model=params.FT_BEST_MODEL_PATH, target=params.FT_TARGET)
    benchmark.head_benchmarks(p_test, device)
    _, num_inputs = in_test.shape
    manager = PortfolioManager(device, num_inputs, mode=enums.Mode.TEST)
    manager.fis.load_genome(params.LAST_MD_GENOME_PATH)
    # manager.fis.load_genome(params.BEST_SR_GENOME_PATH)
    # manager.fis.load_genome(params.BEST_MD_GENOME_PATH)
    # agg = AggregationLayer(device, num_inputs, mode=enums.Mode.TEST)
    for inp, price in zip(in_test, p_test):
        manager.forward(inp, price)
        print(manager.print_portfolio(price))
        # agg.decide(inp, price)
        # print(agg.manager.print_portfolio(price))
    manager.eval()
    # agg.manager.eval()
    benchmark.head_benchmarks(p_test, device)
        



if __name__ == '__main__':
    main()