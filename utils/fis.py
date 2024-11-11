import time
import itertools

import numpy as np
import torch

import utils.helpers as helpers
import utils.evo as evo
import utils.params as params

class GenFIS:
    def __init__(self, device, num_inputs, num_outputs=params.NUM_OUTPUTS, num_in_mf=params.NUM_IN_MF, num_out_mf=params.NUM_OUT_MF, mutation_rate=params.MUTATION_RATE, rule_operator=lambda x: max(x)):
        self.device = device

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_in_mfs = num_in_mf
        self.num_out_mfs = num_out_mf
        self.mutation_rate = mutation_rate
        self.rule_op = rule_operator
        
        self.in_mf_params = torch.rand((num_inputs, num_in_mf), device=device)
        self.out_mf_params = torch.rand((num_outputs, num_out_mf), device=device)
        self.rules = torch.randint(0, self.num_out_mfs, (num_outputs, num_inputs * num_in_mf), dtype=torch.int16, device=device)
        self.mask = torch.randint(0, 2, (num_outputs, num_inputs * num_in_mf), dtype=torch.int16, device=device)
        self.rule_op = rule_operator
        
        self.num_rules = num_inputs*num_in_mf
        
        self.set_in_mfs()
        self.set_out_mfs()
        self.load_consequences()
        
    def randomize(self):
        self.in_mf_params = torch.rand((self.num_inputs, self.num_in_mf), device=self.device)
        self.out_mf_params = torch.rand((self.num_outputs, self.num_out_mf), device=self.device)
        self.rules = torch.randint(0, self.num_out_mfs, (self.num_outputs, self.num_inputs * self.num_in_mf), dtype=torch.int16, device=self.device)
        self.mask = torch.randint(0, 2, (self.num_outputs, self.num_inputs * self.num_in_mf), dtype=torch.int16, device=self.device)
        
    def set_in_mf_params(self, in_mf_params):
        assert in_mf_params.shape == self.in_mf_params.shape
        for i, p in enumerate(in_mf_params):
            in_mf_params[i] = np.sort(p)
        self.in_mf_params = torch.Tensor(in_mf_params)
        self.set_in_mfs()
        
    def set_out_mf_params(self, out_mf_params):
        assert out_mf_params.shape == self.out_mf_params.shape
        self.out_mf_params = torch.Tensor(out_mf_params)
        self.set_out_mfs()
        
    def set_rules(self, rules):
        assert rules.shape == self.rules.shape, (f"{rules.shape} != {self.rules.shape}")
        self.rules = rules
        
    def set_mask(self, mask):
        assert mask.shape == self.mask.shape, (f"{mask.shape} != {self.mask.shape}")
        self.mask = mask
        
    def set_in_mfs(self):
        self.in_mfs = [helpers.get_trimfs(params, self.device) for params in self.in_mf_params]
        
    def set_out_mfs(self):
        self.out_mfs = [helpers.get_trimfs(params, self.device) for params in self.out_mf_params]
        
    def load_consequences(self):
        cons = np.zeros((self.num_outputs, self.num_rules), dtype=object)
        for i in range(self.num_outputs):
            for j in range(self.num_rules):
                mf = self.out_mfs[i][self.rules[i][j]]
                cons[i][j] = mf
        self.cons = cons
        
    def fuzzify(self, inputs):
        inputs = inputs.clone().detach().to(self.device)
        fuzz = torch.zeros((self.num_inputs, self.num_in_mfs), device=self.device)
        
        for i in range(self.num_inputs):
            mfs = self.in_mfs[i]
            for j, mf in enumerate(mfs):
                fuzz[i, j] = mf(inputs[i])
        return fuzz
                
    def activate(self, fuzz):
        ra = torch.stack([self.rule_op(torch.tensor(combo, device=self.device)) for combo in itertools.product(*fuzz)], dim=0)
        return ra
    
    def defuzzify(self, ra, silent):
        centroids = torch.zeros(self.num_outputs, device=self.device)
        self.load_consequences()  # This may need adaptation for GPU, depending on its implementation
        for i in range(self.num_outputs):
            centroid = helpers.calculate_all_centroids(self.cons[i], ra, self.mask[i], self.device, num_points=params.NUM_POINTS)
            centroids[i] = centroid
        return centroids
        
    def forward(self, inputs, silent=True):
        assert len(inputs) == self.num_inputs, f"Length of inputs {len(inputs)} does not match indicated length {self.num_inputs}"
        fuzz = self.fuzzify(inputs)
        act = self.activate(fuzz)
        defuzz = self.defuzzify(act, silent)
        return defuzz
        
    def set_genome(self, genome):
        self.in_mf_params = torch.Tensor(genome[0].reshape((self.num_inputs, self.num_in_mfs))).to(self.device)
        self.out_mf_params = torch.Tensor(genome[1].reshape((self.num_outputs, self.num_out_mfs))).to(self.device)
        self.rules = genome[2].reshape((self.num_outputs, self.num_rules))
        self.mask = torch.Tensor(genome[3].reshape((self.num_outputs, self.num_rules))).to(self.device)
        
        self.set_in_mfs()
        self.set_out_mfs()
        self.load_consequences()
        
    def get_genome(self):
        genome = []
        genome.append(self.in_mf_params.flatten())
        genome.append(self.out_mf_params.flatten())
        genome.append(self.rules.flatten())
        genome.append(self.mask.flatten())
        return genome
    
    def print_genome(self):
        genome = self.get_genome()
        print(genome)
    
    def mutate_genome(self):
        genome = self.get_genome()
        for i in range(len(genome)):
            mutation = evo.mutate_gene(genome[i], self.mutation_rate)
            genome[i] = mutation
        self.set_genome(genome)

    def save_genome(self, path=params.BEST_GENOME_PATH):
        genome = self.get_genome()
        helpers.save_genome(genome, path)

    def load_genome(self, path=params.BEST_GENOME_PATH):
        genome = helpers.load_genome(path)
        self.set_genome(genome)


if __name__ == '__main__':
    fis = GenFIS('cuda', 5)