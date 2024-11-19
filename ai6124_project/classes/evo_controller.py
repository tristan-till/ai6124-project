from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import copy

import numpy as np
import torch
from tqdm import tqdm

import ai6124_project.utils.plot as plot
import ai6124_project.utils.helpers as helpers
import ai6124_project.utils.objectives as objectives
import ai6124_project.utils.params as params

from ai6124_project.utils.evo_utils import crossover_genomes
from ai6124_project.classes.manager import PortfolioManager

class EvolutionController:
    def __init__(self, device, num_inputs, 
                 population_size=params.POPULATION_SIZE, num_generations=params.NUM_GENERATIONS, episode_length=params.EPISODE_LENGTH, 
                 elitism_threshold=params.ELITISM_THRESHOLD, objective=lambda x: objectives.cumulative_return(x),
                 best_model_path=params.BEST_GENOME_PATH, last_model_path=params.LAST_GENOME_PATH, plt_path=params.CR_PLOTS):
        self.device = device
        self.num_inputs = num_inputs
        self.population_size = population_size
        self.num_generations = num_generations
        self.episode_length = episode_length
        self.elitism_threshold = elitism_threshold
        self.objective = objective
        
        self.elitist_size = int(np.ceil(population_size * elitism_threshold))
        self.population = self.init_population()

        self.max_reward = -100
        self.top_genome = None

        self.best_model_path = best_model_path
        self.last_model_path = last_model_path
        self.plt_path = plt_path

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            manager = PortfolioManager(device=self.device, num_inputs=self.num_inputs, rule_operator=lambda x: sum(x) / len(x))
            population.append(manager)
        return population
    
    def returns(self):
        returns = torch.tensor([manager.history[-1] for manager in self.population], device=self.device)
        return returns

    def histories(self):
        histories = [manager.get_history() for manager in self.population]
        return histories
    
    def rewards(self):
        histories = self.histories()
        rewards = []
        for history, buys, sells in histories:
            reward = self.objective(history)
            penalty = self.inactivity_penalty(reward, buys, sells)            
            rewards.append(reward - penalty)
        return torch.tensor(rewards, device=self.device)
    
    def inactivity_penalty(self, reward, buys, sells):
        return 0
        cumulative_penalty = 0
        if len(buys) < 1:
            cumulative_penalty += reward * 0.05 + 0.01
        elif len(sells) < 1:
            cumulative_penalty += reward * 0.01 + 0.01
        return cumulative_penalty
            
    def forward(self, inp, price):
        inp_tensor = inp.clone().detach().to(self.device)
        price_tensor = price.clone().detach().to(self.device)
        def manager_forward(manager, inp, price):
            manager.forward(inp, price)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(manager_forward, manager, inp_tensor, price_tensor) for manager in self.population]
            
            for future in as_completed(futures):
                future.result()
    
    def plot_generation(self, prices, img_path):
        prices = prices
        portfolios, buys, sells = [], [], []
        for manager in self.population:
            p, b, s = manager.get_history()
            portfolios.append(p)
            buys.append(b)
            sells.append(s)
        plot.plot_generation(prices.cpu().numpy(), portfolios, buys, sells, img_path)

    def get_elite_genomes(self, rewards):
        top_rewards= sorted(rewards, reverse=True)[:self.elitist_size]
        top_indexes = [rewards.index(r) for r in top_rewards]
        top_genomes = [self.population[i].fis.get_genome() for i in top_indexes].copy()
        return top_genomes
    
    def crossover_pop(self, rewards):
        assert len(rewards) == len(self.population), "Rewards and population size must match"
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        min_reward = min(rewards)
        max_reward = max(rewards)
        fitness = [(reward - min_reward / max_reward - min_reward) for reward in rewards]
        probs = helpers.softmax(fitness)
        num_pairs = int(len(self.population) / 2 - self.elitist_size)
        for i in range(num_pairs):
            pair = np.random.choice(self.population, 2, p=probs, replace=False)
            genome1, genome2 = crossover_genomes(pair[0].fis.get_genome(), pair[1].fis.get_genome())
            self.population[i*2].fis.set_genome(genome1)
            self.population[i*2+1].fis.set_genome(genome2)
            
    def mutate_pop(self):
        for manager in self.population:
            manager.fis.mutate_genome()
            
    def evolve(self, rewards, max_reward, gen_num):
        elite_genomes = copy.deepcopy(self.get_elite_genomes(rewards.cpu().tolist()))
        top_genome = elite_genomes[0]
        
        helpers.save_genome(top_genome, path=f"{gen_num}_{self.last_model_path}")
        if self.max_reward < max_reward.item():
            self.max_reward = max_reward.item()
            self.top_genome = elite_genomes[0]
            helpers.save_genome(self.top_genome, path=self.best_model_path)
        
        self.crossover_pop(rewards.cpu().tolist())
        self.mutate_pop()
        for i in range(self.elitist_size):
            self.population[-i-1].fis.set_genome(elite_genomes[i])
            
    def evo_step(self, inps, prices, gen_num):
        t1 = time.time()
        assert len(inps) == len(prices), f"Input ({len(inps)}) and price length {len(prices)} must match"
        start_idx = torch.randint(0, len(inps) - self.episode_length + 1, (1,)).item()
        seq_inps = inps[start_idx:start_idx + self.episode_length]
        seq_prices = prices[start_idx:start_idx + self.episode_length]
        # seq_inps = inps
        # seq_prices = prices

        progress_bar = tqdm(
            range(self.episode_length),
            desc=f"Generation {gen_num}",
            unit="step",
            leave=True,
            colour="cyan"
        )

        for i in progress_bar:
            self.forward(seq_inps[i], seq_prices[i])
            progress_bar.set_postfix({"Elapsed Time": f"{time.time() - t1:.2f}s"})

        returns = self.returns()        
        rewards = self.rewards()
        max_reward = torch.max(rewards)
        max_returns = max(returns).item()
        initial_cash = params.INITIAL_CASH + params.INITIAL_STOCKS * seq_prices[0].item()
        print(f"Max episode return: {(max_returns):.2f} ({(((max_returns - initial_cash) / initial_cash)*100):.2f}%)")
        print(f"Max episode reward: {(max_reward.item()):.2f}")
        
        helpers.save_genome(self.get_elite_genomes(returns.cpu().tolist())[0], path=f"maxcm_{gen_num}_{self.last_model_path}")
        self.plot_generation(seq_prices, f"{self.plt_path}/gen_{gen_num}.png")
        self.evolve(rewards, max_reward, gen_num)
        
        self.reset()
        t2 = time.time()
        print(f"Episode Duration: {(t2- t1):.2f}")
        
    def train(self, inps, prices):
        print("Starting training")
        if not isinstance(inps, torch.Tensor):
            inps = torch.tensor(inps).to(self.device)
        if not isinstance(prices, torch.Tensor):
            prices = torch.tensor(prices).to(self.device)
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