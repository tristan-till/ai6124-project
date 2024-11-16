import time
import torch
import numpy as np

import utils.params as params
import utils.evo_data as evo_data

from utils.aggregation import AggregationLayer

def main():
    device='cuda'
    from utils.fis import GenFIS
    fis = GenFIS(device, 4)
    fis.load_genome("temp.h5")
    fis.print_genome()
    fis.benchmark_genome()
    fis.print_genome()

if __name__ == '__main__':
    main()