import math
import fuzzylab as fz
import numpy as np
import matplotlib.pyplot as plt

def check_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def trimf(x, params):
    lb = 0.0
    ub = 1.0
    a, b, c = params
    if x < b and a == -math.inf: return 1.0
    if x > b and c == math.inf: return 1.0
    if x < lb or x > ub: return 0.0
    return fz.trimf(x, [a, b, c])[0]

def get_trimfs(params):
    params = np.sort(params)
    params = np.insert(params, 0, -math.inf)
    params = np.append(params, math.inf)
    mfs = []
    for i in range(len(params)-2):
        a, b, c = params[i], params[i+1], params[i+2]
        def mf(x, a=a, b=b, c=c):
            return trimf(x, [a, b, c])
        mfs.append(mf)
    return mfs

def calculate_cut_area_and_centroid(x, mf_values, rule_activation):
    cut_values = np.minimum(mf_values, rule_activation)
    area = np.trapezoid(cut_values, x)
    centroid = np.trapezoid(x * cut_values, x) / area if area > 0 else 0
    return area, centroid

def get_centroids(out_mfs, rule_activations, rule_mask, silent=True):
    x = np.linspace(0, 1, 500)
    total_area = 0
    weighted_centroid_sum = 0
    if not silent:
        plt.figure(figsize=(10, 6))
    for i, (mf, activation, mask) in enumerate(zip(out_mfs, rule_activations, rule_mask)):
        if not mask:
            activation = 0
        mf_values = np.array([mf(x) for x in x])
        area, centroid = calculate_cut_area_and_centroid(x, mf_values, activation)
        
        total_area += area
        weighted_centroid_sum += area * centroid
        
        cut_values = np.minimum(mf_values, activation)
        if not silent:
            plt.plot(x, mf_values, label=f'MF {i+1}')
            plt.fill_between(x, 0, cut_values, alpha=0.3)
        
    overall_centroid = weighted_centroid_sum / total_area if total_area != 0 else 0
    if not silent:
        plt.axvline(overall_centroid, color='r', linestyle='--', label=f'Overall Centroid = {overall_centroid:.3f}')

        plt.title("Membership Functions with Rule Activations and Combined Centroid")
        plt.xlabel("x")
        plt.ylabel("Membership Degree")
        plt.legend()
    
        plt.show()
        plt.close()
    return overall_centroid

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)