import itertools

import numpy as np

import utils.helpers as helpers
import utils.evo as evo

class GenFIS:
    def __init__(self, num_inputs, num_outputs, num_in_mf, num_out_mf, mutation_rate=0.1, rule_operator=lambda x: max(x)):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_in_mfs = num_in_mf
        self.num_out_mfs = num_out_mf
        self.mutation_rate = mutation_rate
        self.rule_op = rule_operator
        
        self.in_mfs = np.zeros((num_inputs, num_in_mf), dtype=object)
        self.out_mfs = np.zeros((num_outputs, num_out_mf), dtype=object)
        self.in_mf_params = np.zeros((num_inputs, num_in_mf))
        self.out_mf_params = np.zeros((num_outputs, num_out_mf))
        self.rules = np.zeros((num_outputs, num_inputs*num_in_mf), dtype=np.int16)
        self.mask = np.zeros((num_outputs, num_inputs*num_in_mf), dtype=np.int16)
        self.rule_op = rule_operator
        
        self.num_rules = num_inputs*num_in_mf
        
        self.set_in_mfs()
        self.set_out_mfs()
        self.load_consequences()
        
    def randomize(self):
        in_mf_params = np.random.rand(self.num_inputs, self.num_in_mfs)
        out_mf_params = np.random.rand(self.num_outputs, self.num_out_mfs)
        rules = np.random.choice([i for i in range(self.num_out_mfs)], size=(self.num_outputs, self.num_inputs*self.num_in_mfs))
        mask = np.random.choice([0, 1], size=(self.num_outputs, self.num_inputs*self.num_in_mfs))
        
        self.set_in_mf_params(in_mf_params)
        self.set_out_mf_params(out_mf_params)
        self.set_rules(rules)
        self.set_mask(mask)
        
    def set_in_mf_params(self, in_mf_params):
        assert in_mf_params.shape == self.in_mf_params.shape
        for i, p in enumerate(in_mf_params):
            in_mf_params[i] = np.sort(p)
        self.in_mf_params = in_mf_params
        self.set_in_mfs()
        
    def set_out_mf_params(self, out_mf_params):
        assert out_mf_params.shape == self.out_mf_params.shape
        self.out_mf_params = out_mf_params
        self.set_out_mfs()
        
    def set_rules(self, rules):
        assert rules.shape == self.rules.shape, (f"{rules.shape} != {self.rules.shape}")
        self.rules = rules
        
    def set_mask(self, mask):
        assert mask.shape == self.mask.shape, (f"{mask.shape} != {self.mask.shape}")
        self.mask = mask
        
    def set_in_mfs(self):
        self.in_mfs = [helpers.get_trimfs(params) for params in self.in_mf_params]
        
    def set_out_mfs(self):
        self.out_mfs = [helpers.get_trimfs(params) for params in self.out_mf_params]
        
    def load_consequences(self):
        cons = np.zeros((self.num_outputs, self.num_rules), dtype=object)
        for i in range(self.num_outputs):
            for j in range(self.num_rules):
                mf = self.out_mfs[i][self.rules[i][j]]
                cons[i][j] = mf
        self.cons = cons
        
    def fuzzify(self, inputs):
        fuzz = np.zeros((self.num_inputs, self.num_in_mfs))
        for i, inp in enumerate(inputs):
            mfs = self.in_mfs[i]
            for j, mf in enumerate(mfs):
                fuzz[i][j] += mf(inp)
        return fuzz
                
    def activate(self, fuzz):
        ra = np.array([
            self.rule_op(combo) for combo in itertools.product(*fuzz)
        ])
        return ra
    
    def defuzzify(self, ra, silent):
        centroids = np.zeros(self.num_outputs)
        self.load_consequences()
        for i in range(self.num_outputs):
            centroid = helpers.get_centroids(self.cons[i], ra, self.mask[i], silent=silent)
            centroids[i] = centroid
        return centroids
        
    def forward(self, inputs, silent=True):
        assert len(inputs) == self.num_inputs
        fuzz = self.fuzzify(inputs)
        act = self.activate(fuzz)
        defuzz = self.defuzzify(act, silent)
        return defuzz
        
    def set_genome(self, genome):
        self.in_mf_params = np.array(genome[0]).reshape((self.num_inputs, self.num_in_mfs))
        self.out_mf_params = np.array(genome[1]).reshape((self.num_outputs, self.num_out_mfs))
        self.rules = np.array(genome[2]).reshape((self.num_outputs, self.num_rules))
        self.mask = np.array(genome[3]).reshape((self.num_outputs, self.num_rules))
        
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