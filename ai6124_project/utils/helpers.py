import random
import requests

import fuzzylab as fz
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import os

import ai6124_project.utils.params as params

def set_deterministic(seed=params.SEED):
    """
    Set random seeds and configure PyTorch for deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_weights(url, out_path):
    response = requests.get(url, stream=True)
    output_path = f"weights/{out_path}"
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"File downloaded successfully and saved as {output_path}.")

def download_h5_file(url, out_path):
    """
    Downloads an HDF5 file from the given URL and validates it.

    Parameters:
        url (str): The URL of the HDF5 file.
        out_path (str): The relative path to save the downloaded file.

    Returns:
        str: Path to the downloaded file.

    Raises:
        ValueError: If the download fails or the file is invalid.
    """
    # Define full output path
    output_path = f"weights/{out_path}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Stream download the file with proper headers
    headers = {"User-Agent": "Mozilla/5.0"}  # Add headers if necessary
    response = requests.get(url, stream=True, headers=headers)

    # Check if the request was successful
    if response.status_code != 200:
        raise ValueError(f"Failed to download file from {url}. Status code: {response.status_code}")

    # Save the file
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    # Validate that the file is a valid HDF5
    try:
        with h5py.File(output_path, "r") as _:
            print(f"Downloaded and validated HDF5 file saved at {output_path}.")
    except OSError as e:
        os.remove(output_path)  # Remove invalid file
        raise ValueError(f"Downloaded file is not a valid HDF5 file: {e}")
    
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

def centroids(out_mf, out_mf_linspace, ra, consequences, device):
    x_range = torch.linspace(0, 1, params.NUM_POINTS, device=device).to(device)
    centroids = torch.zeros((params.NUM_OUTPUTS)).to(device)
    for i, cons in enumerate(consequences):
        out_agg = torch.zeros((params.NUM_POINTS)).to(device)
        for j, c in enumerate(cons):
            mf = out_mf[i][c]
            cut = mf(ra[j])
            cut_linspace = torch.ones((params.NUM_POINTS)).to(device) * cut 
            out_agg = torch.max(torch.fmin(out_mf_linspace[i][c], cut_linspace), out_agg).to(device)
        out_agg = torch.pow(out_agg, params.POW)
        numerator = torch.trapz(x_range * out_agg, x_range)
        denominator = torch.trapz(out_agg, x_range)
        centroid = numerator / denominator if denominator != 0 else 0.0
        centroids[i] = centroid
    return centroids

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