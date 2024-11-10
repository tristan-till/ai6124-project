import math

import numpy as np

def crossover_gene(gene1, gene2):
    gene1 = gene1.flatten()
    gene2 = gene2.flatten()
    assert len(gene1) == len(gene2)
    seq_len = len(gene1)
    seq_choices = np.arange(1, seq_len-1)
    cut1 = np.random.choice(seq_choices)
    seq_choices = np.setdiff1d(seq_choices, [cut1-1, cut1, cut1+1])
    if len(seq_choices) < 1:
        cut2 = seq_len - 1
    else:
        cut2 = np.random.choice(seq_choices)
    cut1, cut2 = min(cut1, cut2), max(cut1, cut2)
    
    seq_a1 = gene1[:cut1]
    seq_a2 = gene2[:cut1]
    seq_b1 = gene1[cut1:cut2]
    seq_b2 = gene2[cut1:cut2]
    seq_c1 = gene1[cut2:]
    seq_c2 = gene2[cut2:]
    gene1 = np.concat([seq_a1, seq_b2, seq_c1])
    gene2 = np.concat([seq_a2, seq_b1, seq_c2])
    return gene1, gene2

def crossover_genomes(genome1, genome2):
    assert len(genome1) == len(genome2), "Genomes must be of equal length"
    num_genes = len(genome1)
    for i in range(num_genes):
        assert len(genome1[i].flatten()) == len(genome2[i].flatten()), "Genomes must be of equal length"
        genome1[i], genome2[i] = crossover_gene(genome1[i], genome2[i])
    return genome1, genome2


def mutate_int_gene(gene, indices):
    max_val = max(np.append(gene, 1))
    muts = np.random.choice(np.arange(1, max_val+1), size=len(indices), replace=True)
    for i, ind in enumerate(indices):
        gene[ind] = abs(gene[ind] - muts[i])
    return gene

def mutate_float_gene(gene, indices):
    muts = np.random.uniform(0, 1, size=len(indices))
    for i, ind in enumerate(indices):
        gene[ind] = abs(gene[ind] - muts[i])
    return gene
    
def mutate_gene(gene, mutation_rate=0.1):
    gene = gene.flatten()
    num = math.ceil(len(gene) * mutation_rate)
    mut_indices = np.random.choice(np.arange(0, len(gene)), size=num, replace=False)
    if gene.dtype == np.int64:
        return mutate_int_gene(gene, mut_indices)
    elif gene.dtype == np.float64:
        return mutate_float_gene(gene, mut_indices)
    else:
        raise TypeError("Unrecognized gene type {}".format(gene.dtype))