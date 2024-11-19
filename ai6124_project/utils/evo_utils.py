import math

import numpy as np
import torch

def crossover_gene(gene1: torch.Tensor, gene2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gene1_flat = gene1.flatten()
    gene2_flat = gene2.flatten()
    device = gene1.device
    
    seq_len = gene1_flat.size(0)
    assert seq_len == gene2_flat.size(0), "Genes must be of equal length"
    
    seq_choices = torch.arange(1, seq_len-1, device=device)
    
    cut1 = seq_choices[torch.randint(len(seq_choices), (1,), device=device)]
    
    mask = torch.ones_like(seq_choices, dtype=torch.bool)
    mask[max(0, cut1-1):min(len(mask), cut1+2)] = False
    valid_choices = seq_choices[mask]
    
    if len(valid_choices) < 1:
        cut2 = torch.tensor(seq_len - 1, device=device)
    else:
        cut2 = valid_choices[torch.randint(len(valid_choices), (1,), device=device)]
    
    cut1, cut2 = torch.min(cut1, cut2), torch.max(cut1, cut2)
    
    new_gene1 = torch.empty_like(gene1_flat)
    new_gene2 = torch.empty_like(gene2_flat)
    
    new_gene1[:cut1] = gene1_flat[:cut1]
    new_gene1[cut1:cut2] = gene2_flat[cut1:cut2]
    new_gene1[cut2:] = gene1_flat[cut2:]
    
    new_gene2[:cut1] = gene2_flat[:cut1]
    new_gene2[cut1:cut2] = gene1_flat[cut1:cut2]
    new_gene2[cut2:] = gene2_flat[cut2:]
    
    return new_gene1.reshape(gene1.shape), new_gene2.reshape(gene2.shape)

def crossover_genomes(genome1: list[torch.Tensor], genome2: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    assert len(genome1) == len(genome2), "Genomes must be of equal length"
    
    if len(genome1) <= 10:
        results = [crossover_gene(g1, g2) for g1, g2 in zip(genome1, genome2)]
        new_genome1 = [res[0] for res in results]
        new_genome2 = [res[1] for res in results]
    else:
        new_genome1 = []
        new_genome2 = []
        for g1, g2 in zip(genome1, genome2):
            crossed_g1, crossed_g2 = crossover_gene(g1, g2)
            new_genome1.append(crossed_g1)
            new_genome2.append(crossed_g2)
    
    return new_genome1, new_genome2

def mutate_int_gene(gene, indices):
    max_val = max(gene.max().item(), 1)
    muts = torch.randint(1, max_val + 1, (len(indices),), 
                        device=gene.device, 
                        dtype=gene.dtype)
    gene[indices] = torch.abs(gene[indices] - muts)
    return gene

def mutate_float_gene(gene, indices):
    muts = torch.rand(len(indices), 
                     device=gene.device, 
                     dtype=gene.dtype)
    gene[indices] = torch.abs(gene[indices] - muts)
    return gene

def mutate_gene(gene, mutation_rate=0.1):
    gene = gene.flatten()
    num_mutations = math.ceil(len(gene) * mutation_rate)
    mut_indices = torch.randperm(len(gene), device=gene.device)[:num_mutations]
    if gene.dtype in [torch.int16, torch.int32, torch.int64, torch.long, torch.short]:
        return mutate_int_gene(gene, mut_indices)
    elif gene.dtype in [torch.float16, torch.float32, torch.float64, torch.float]:
        return mutate_float_gene(gene, mut_indices)
    else:
        raise TypeError(f"Unrecognized gene type {gene.dtype}")