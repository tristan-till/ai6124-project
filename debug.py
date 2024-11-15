import time
import torch

import utils.params as params
import utils.evo_data as evo_data

from utils.aggregation import AggregationLayer

def main():
    device='cuda'
    from utils.fis import GenFIS
    fis = GenFIS(device, 4)
    fis.load_genome("temp.h5")
    fis.explain()
    
from concurrent.futures import ThreadPoolExecutor, as_completed

def exec_in_parallel(f, it):
     with ThreadPoolExecutor() as executor:
        res = []
        futures = [executor.submit(f, x) for x in it]
        for future in as_completed(futures):
            res.append(future.result())
        return res

if __name__ == '__main__':
    main()