import torch
from collections import Counter

from utils.manager import PortfolioManager
import utils.params as params
import utils.enums as enums

class AggregationLayer():
    def __init__(self, device, num_inputs, mode=enums.Mode.TRAIN):
        self.device = device
        self.mode = mode
        self.managers = [PortfolioManager(device=device, num_inputs=num_inputs, mode=mode) for i in range(params.NUM_OBJECTIVES)]
        self.manager = PortfolioManager(device=device, num_inputs=0, mode=mode)
        self.init_managers()

    def init_managers(self):
        self.managers[0].fis.load_genome(params.BEST_CR_GENOME_PATH)
        self.managers[1].fis.load_genome(params.BEST_SR_GENOME_PATH)
        self.managers[2].fis.load_genome(params.BEST_MD_GENOME_PATH)

    def forward(self, inputs, price):
        decisions = torch.zeros((len(self.managers), params.NUM_OUTPUTS))
        for i, manager in enumerate(self.managers):
            actions = manager.forward(inputs, price, act=False)
            decisions[i] = actions
        return decisions

    def decide(self, inputs, price):
        decisions = self.forward(inputs, price)
        vote, activation = self.vote(decisions)
        actions = torch.Tensor([0, 0, 0]).to(self.device)
        actions[vote] = activation
        self.manager.act(actions, price)
        self.sync_liquidity()
        
    def sync_liquidity(self):
        for manager in self.managers:
            manager.cash = self.manager.cash
            manager.num_stocks = self.manager.num_stocks

    def vote(self, decisions):
        primary_votes = torch.argmax(decisions, dim=1)
        n = len(primary_votes)
        vote_counts = torch.bincount(primary_votes)
        majority_vote = torch.argmax(vote_counts)
        majority_count = vote_counts[majority_vote]
        if majority_count > n / 2:
            positive_voters = decisions[:, majority_vote] > 0
            activation_value = decisions[positive_voters, majority_vote].mean()
            return majority_vote.item(), activation_value.item()
        return 1, 1.0
