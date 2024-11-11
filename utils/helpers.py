import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import fuzzylab as fz
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch

import utils.params as params

def check_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def trimf(x, params, device):
    lb = 0.0
    ub = 1.0
    a, b, c = params
    if x < b and torch.isinf(a): return torch.tensor(1.0, device=device)
    if x > b and torch.isinf(c): return torch.tensor(1.0, device=device)
    if x < lb or x > ub: return torch.tensor(0.0, device=device)
    return torch.fmax(torch.tensor(0.0, device=device), torch.fmin((x - a) / (b - a), (c - x) / (c - b)))

def get_trimfs(params, device):
    params = torch.sort(params).values  # Sort params
    params = torch.cat((torch.tensor([-float('inf')], device=device), params, torch.tensor([float('inf')], device=device)))
    mfs = []
    for i in range(len(params)-2):
        a, b, c = params[i], params[i+1], params[i+2]
        def mf(x, a=a, b=b, c=c):
            return trimf(x, [a, b, c], device)
        mfs.append(mf)
    return mfs

def calculate_cut_area_and_centroid(x, mf_values, rule_activation, device):
    t1 = time.time()
    cut_values = torch.minimum(mf_values, rule_activation)
    t2 = time.time()
    area = torch.trapz(cut_values, x)
    t3 = time.time()
    centroid = torch.trapz(x * cut_values, x) / area if area > 0 else torch.tensor(0.0, device=device)
    t4 = time.time()
    print(f"Calc took {(t4 - t1):.2f}, T1: {(t2 - t1):.2f}, T2: {(t3 - t2):.2f}, T3: {(t4 - t3):.2f}")
    return area, centroid

def get_centroids(out_mfs, rule_activations, rule_mask, device):
    x = torch.linspace(0, 1, 100, device=device)
    total_area = torch.tensor(0.0, device=device)
    weighted_centroid_sum = torch.tensor(0.0, device=device)
    for i, (mf, activation, mask) in enumerate(zip(out_mfs, rule_activations, rule_mask)):
        t1 = time.time()
        if not mask:
            continue
        t2 = time.time()
        def get_mf_val(mf, x):
            return mf(x)
        mf_values = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(get_mf_val, mf, xi) for xi in x]
            for future in as_completed(futures):
                mf_values.append(future.result())
        mf_values = torch.stack(mf_values)
        t3 = time.time()
        area, centroid = calculate_cut_area_and_centroid(x, mf_values, activation, device)
        t4 = time.time()
        total_area += area
        weighted_centroid_sum += area * centroid   
        t5 = time.time()
        print(f"Loop took {(t5 - t1):.2f}, T1: {(t2 - t1):.2f}, T2: {(t3 - t2):.2f}, T3: {(t4 - t3):.2f}, T4: {(t5 - t4):.2f}")
    overall_centroid = weighted_centroid_sum / total_area if total_area != 0 else 0
    return overall_centroid.item()

def get_centroids_and_plot(out_mfs, rule_activations, rule_mask, device, silent=True):
    x = torch.linspace(0, 1, 500, device=device)
    total_area = torch.tensor(0.0, device=device)
    weighted_centroid_sum = torch.tensor(0.0, device=device)

    if not silent:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
    
    for i, (mf, activation, mask) in enumerate(zip(out_mfs, rule_activations, rule_mask)):
        if not mask:
            activation = torch.tensor(0.0, device=device)
        
        mf_values = torch.stack([mf(xi) for xi in x])
        area, centroid = calculate_cut_area_and_centroid(x, mf_values, activation, device)
        
        total_area += area
        weighted_centroid_sum += area * centroid
        
        cut_values = torch.minimum(mf_values, activation)
        if not silent:
            plt.plot(x.cpu().numpy(), mf_values.cpu().numpy(), label=f'MF {i+1}')
            plt.fill_between(x.cpu().numpy(), 0, cut_values.cpu().numpy(), alpha=0.3)
    
    overall_centroid = weighted_centroid_sum / total_area if total_area != 0 else 0
    if not silent:
        plt.axvline(overall_centroid.cpu().item(), color='r', linestyle='--', label=f'Overall Centroid = {overall_centroid:.3f}')
        plt.title("Membership Functions with Rule Activations and Combined Centroid")
        plt.xlabel("x")
        plt.ylabel("Membership Degree")
        plt.legend()
        plt.show()
        plt.close()
    
    return overall_centroid.item()

def calculate_all_centroids(out_mfs, rule_activations, rule_mask, device, num_points=params.NUM_POINTS):
    x = torch.linspace(0, 1, num_points, device=device)
    
    valid_indices = [i for i, mask in enumerate(rule_mask) if mask]
    valid_mfs = [out_mfs[i] for i in valid_indices]
    valid_activations = rule_activations[valid_indices]
    
    if not valid_mfs:
        return 0.0    
    mf_values = evaluate_all_mfs(valid_mfs, x, device)
    cut_values = torch.minimum(mf_values, valid_activations.unsqueeze(1))    
    areas = torch.trapz(cut_values, x, dim=1)
    x_expanded = x.unsqueeze(0).expand(len(valid_mfs), -1)
    centroids = torch.trapz(x_expanded * cut_values, x, dim=1) / torch.clamp(areas, min=1e-10)
    total_area = areas.sum()
    if total_area > 0:
        overall_centroid = (areas * centroids).sum() / total_area
    else:
        overall_centroid = torch.tensor(0.0, device=device)
    return overall_centroid.item()

def evaluate_all_mfs(mfs, x, device):
    def evaluate_single_mf(mf, x_tensor):
        def get_mf_val(mf, x):
            return mf(x)
        mf_values = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(get_mf_val, mf, x) for x in x_tensor]
            for future in as_completed(futures):
                mf_values.append(future.result())
        return torch.stack(mf_values)
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda mf: evaluate_single_mf(mf, x),
            mfs
        ))
    return torch.stack(results)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def save_genome(genome, path):
    with h5py.File(f"weights/{path}", 'w') as file:
        for i, arr in enumerate(genome):
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            file.create_dataset(f"gene_{i}", data=arr)

def load_genome(path):
    with h5py.File(f"weights/{path}", 'r') as file:
        genome = [file[f"gene_{i}"][:] for i in range(params.NUM_GENES)]
    return genome