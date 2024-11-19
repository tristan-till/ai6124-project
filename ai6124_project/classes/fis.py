import time
import itertools

import numpy as np
import torch

import utils.helpers as helpers
import ai6124_project.utils.evo_utils as evo_utils
import utils.params as params
import utils.benchmark as benchmark

class GenFIS:
    def __init__(self, device, num_inputs, mutation_rate=params.MUTATION_RATE, rule_operator=lambda x: max(x)):
        self.device = device

        self.num_inputs = num_inputs
        self.num_outputs = params.NUM_OUTPUTS
        self.num_in_mfs = params.NUM_IN_MF
        self.num_out_mfs = params.NUM_OUT_MF
        self.num_rules = params.NUM_RULES
        self.mutation_rate = mutation_rate
        self.rule_op = rule_operator
        
        self.in_mf_params = torch.rand((num_inputs, self.num_in_mfs), device=device)
        self.out_mf_params = torch.rand((self.num_outputs, self.num_out_mfs), device=device)
        self.rules = torch.randint(0, self.num_in_mfs, (self.num_rules, self.num_inputs), dtype=torch.int16, device=device)
        self.consequences = torch.randint(0, self.num_out_mfs, (self.num_outputs, self.num_rules), dtype=torch.int16, device=device)

        self.rule_op = rule_operator
                
        self.set_in_mfs()
        self.set_out_mfs()
        
    def set_in_mf_params(self, in_mf_params):
        assert in_mf_params.shape == self.in_mf_params.shape
        for i, p in enumerate(in_mf_params):
            in_mf_params[i] = np.sort(p)
        self.in_mf_params = torch.Tensor(in_mf_params).to(self.device)
        self.set_in_mfs()
        
    def set_out_mf_params(self, out_mf_params):
        assert out_mf_params.shape == self.out_mf_params.shape
        self.out_mf_params = torch.Tensor(out_mf_params)
        self.set_out_mfs()
        
    def set_rules(self, rules):
        assert rules.shape == self.rules.shape, (f"{rules.shape} != {self.rules.shape}")
        self.rules = rules
        
    def set_consequences(self, consequences):
        assert consequences.shape == self.consequences.shape, (f"{consequences.shape} != {self.consequences.shape}")
        self.consequences = consequences
        
    def set_in_mfs(self):
        self.in_mfs = [helpers.get_trimfs(params, self.device) for params in self.in_mf_params]
        
    def set_out_mfs(self):
        self.out_mfs = [helpers.get_trimfs(params, self.device) for params in self.out_mf_params]
        
    def fuzzify(self, inputs):
        inputs = inputs.clone().detach().to(self.device)
        fuzz = torch.zeros((self.num_inputs, self.num_in_mfs), device=self.device)
        
        for i in range(self.num_inputs):
            mfs = self.in_mfs[i]
            for j, mf in enumerate(mfs):
                fuzz[i, j] = mf(inputs[i])
        return fuzz
                
    def activate(self, fuzz):
        ra = torch.zeros((self.num_rules,)).to(self.device)
        for i, rule in enumerate(self.rules):
            activation = 0.0
            for j, x in enumerate(rule):
                activation += fuzz[j][x]
            ra[i] = activation / self.num_inputs
        return ra
    
    def defuzzify(self, ra):
        centroids = helpers.centroids(self.out_mfs, ra, self.consequences, self.device)
        return centroids
        
    def forward(self, inputs):
        assert len(inputs) == self.num_inputs, f"Length of inputs {len(inputs)} does not match indicated length {self.num_inputs}"
        fuzz = self.fuzzify(inputs)
        act = self.activate(fuzz)
        defuzz = self.defuzzify(act)
        return defuzz
        
    def set_genome(self, genome):
        self.in_mf_params = torch.Tensor(genome[0].reshape((self.num_inputs, self.num_in_mfs))).to(self.device)
        self.out_mf_params = torch.Tensor(genome[1].reshape((self.num_outputs, self.num_out_mfs))).to(self.device)
        self.rules = torch.Tensor(genome[2].reshape((self.num_rules, self.num_inputs))).int().to(self.device)
        self.consequences = torch.Tensor(genome[3].reshape((self.num_outputs, self.num_rules))).int().to(self.device)
        
        self.set_in_mfs()
        self.set_out_mfs()
        
    def get_genome(self):
        genome = []
        genome.append(self.in_mf_params.flatten())
        genome.append(self.out_mf_params.flatten())
        genome.append(self.rules.flatten())
        genome.append(self.consequences.flatten())
        return genome
    
    def print_genome(self):
        genome = self.get_genome()
        print(genome)
    
    def mutate_genome(self):
        genome = self.get_genome()
        for i in range(len(genome)):
            mutation = evo_utils.mutate_gene(genome[i], self.mutation_rate)
            genome[i] = mutation
        self.set_genome(genome)

    def save_genome(self, path=params.BEST_GENOME_PATH):
        genome = self.get_genome()
        helpers.save_genome(genome, path)

    def load_genome(self, path=params.BEST_GENOME_PATH):
        genome = helpers.load_genome(path)
        self.set_genome(genome)
    
    def benchmark_genome(self):
        genome = benchmark.benchmark_genome()
        self.set_genome(genome)

    def explain(self):
        rules = self.rules
        cons = torch.rot90(self.consequences, 3).flip(1)
        text = ""
        ling_term = ["small", "medium", "large"]
        out_terms = ["sell", "hold", "buy"]
        for i in range(self.num_rules):
            text += f"Rule {i+1}: If "
            rs = [f"x{j+1} is {ling_term[r.item()]}" for j, r in enumerate(rules[i])]
            text += " AND ".join(rs)
            text += " then "
            cs = [f"{out_terms[j]} is {ling_term[c.item()]}" for j, c in enumerate(cons[i])]
            text += " AND ".join(cs)
            text += "\n"
        print(text)


if __name__ == '__main__':
    fis = GenFIS('cuda', 5)