import numpy as np

import utils.plot as plot
import utils.helpers as helpers

from utils.evo import crossover_genomes
from utils.manager import PortfolioManager

class EvolutionController:
    def __init__(self, population_size=10, num_generations=10, elitism_threshold=0.1):
        self.population_size = population_size
        self.num_generations = num_generations
        self.elitism_threshold = elitism_threshold
        
        self.elitist_size = int(np.ceil(population_size * elitism_threshold))
        self.population = self.init_population()

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            manager = PortfolioManager(100, 0.02, 5, 2, 3, 3, mutation_rate=0.1, rule_operator=lambda x: sum(x) / len(x))
            population.append(manager)
        return population
    
    def rewards(self, price):
        rewards = []
        for manager in self.population:
            reward = manager.get_value(price)
            if len(manager.buys) < 1:
                reward -= 25
            elif len(manager.sells) < 1:
                reward -= 5
            rewards.append(reward)
        return rewards
    
    def crossover_pop(self, rewards):
        assert len(rewards) == len(self.population), "Rewards and population size must match"
        reward_sum = sum(rewards)
        fitness = [r / reward_sum for r in rewards]
        props = helpers.softmax(fitness)
        num_pairs = int(len(self.population) / 2 - self.elitist_size)
        for i in range(num_pairs):
            pair = np.random.choice(self.population, 2, p=props, replace=False)
            genome1, genome2 = crossover_genomes(pair[0].fis.get_genome(), pair[1].fis.get_genome())
            self.population[i*2].fis.set_genome(genome1)
            self.population[i*2+1].fis.set_genome(genome2)
            
    def mutate_pop(self):
        for i, manager in enumerate(self.population):
            manager.fis.mutate_genome()
            
    def forward(self, inp, price):
        for manager in self.population:
            manager.forward(inp, price)
            
    def get_elite_genomes(self, rewards):
        top_rewards= sorted(rewards, reverse=True)[:self.elitist_size]
        top_indexes = [rewards.index(r) for r in top_rewards]
        top_genomes = [self.population[i].fis.get_genome() for i in top_indexes].copy()
        return top_genomes
    
    def plot_generation(self, prices, img_path):
        prices = prices
        portfolios, buys, sells = [], [], []
        for manager in self.population:
            p, b, s = manager.get_history()
            portfolios.append(p)
            buys.append(b)
            sells.append(s)
        plot.plot_generation(prices, portfolios, buys, sells, img_path)
            
            
    def evo_step(self, inps, prices, gen_num):
        assert len(inps) == len(prices), "Input and price length must match"
        episode_length = len(inps)
        for i in range(episode_length):
            self.forward(inps[i], prices[i])
        rewards = self.rewards(prices[-1])
        print("Max episode reward: ", max(rewards))
        self.plot_generation(prices, f"gen_{gen_num}")
        elite_genomes = self.get_elite_genomes(rewards)
        self.crossover_pop(rewards)
        self.mutate_pop()
        for i in range(self.elitist_size):
            self.population[-i-1].fis.set_genome(elite_genomes[i])
        self.reset()
        
    def train(self, inps, prices):
        print("Starting training")
        for i in range(self.num_generations):
            print(f"Epoch {i+1} / {self.num_generations}")
            self.evo_step(inps, prices, i+1)
            
    def reset(self):
        for manager in self.population:
            manager.reset()
    
def main():
    controller = EvolutionController(4, 5)
    prices = [1, 1.2, 1.4, 1.6, 1.8]
    time_series = [
        [0.25, 0.8, 0.9, 0.5],
        [0.5, 0.8, 0.7, 0.3],
        [0.75, 0.8, 0.5, 0.5],
        [1.0, 0.8, 0.3, 0.7],
        [1.0, 0.8, 0.3, 0.7],
    ]
    controller.train(time_series, prices)
    

if __name__ == '__main__':
    main()