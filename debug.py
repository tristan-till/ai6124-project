import time
import torch
import numpy as np

import utils.data as data
import utils.params as params
import utils.evo_data as evo_data
import utils.benchmark as benchmark
import utils.enums as enums

from utils.aggregation import AggregationLayer
from utils.manager import PortfolioManager

def main():
    device='cuda'
    dataset, num_features = data.prepare_data()
    print(num_features)
    
        



if __name__ == '__main__':
    main()