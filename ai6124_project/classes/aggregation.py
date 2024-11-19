import torch

from ai6124_project.classes.manager import PortfolioManager
import ai6124_project.utils.params as params
import ai6124_project.utils.enums as enums

class AggregationLayer():
    def __init__(self, device, num_inputs, mode=enums.Mode.TRAIN, genome_paths=[params.BEST_CR_GENOME_PATH, params.BEST_SR_GENOME_PATH, params.BEST_MD_GENOME_PATH]):
        self.device = device
        self.mode = mode
        self.genome_paths = genome_paths
        self.managers = [PortfolioManager(device=device, num_inputs=num_inputs, mode=mode) for i in range(len(genome_paths))]
        self.manager = PortfolioManager(device=device, num_inputs=0, mode=mode)
        self.init_managers()

    def init_managers(self):
        for i, path in enumerate(self.genome_paths):
            self.managers[i].fis.load_genome(path)

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
        unanimous_mask = (decisions == decisions.max(dim=1, keepdim=True).values).sum(dim=1) == 1
        filtered_decisions = decisions[unanimous_mask]
        if filtered_decisions.size(0) == 0:
            return 1, 1.0
        primary_votes = torch.argmax(filtered_decisions, dim=1)
        n = len(primary_votes)
        vote_counts = torch.bincount(primary_votes, minlength=decisions.size(1))
        majority_vote = torch.argmax(vote_counts)
        majority_count = vote_counts[majority_vote]
        if majority_count > n / 2:
            positive_voters = filtered_decisions[:, majority_vote] > 0
            activation_value = filtered_decisions[positive_voters, majority_vote].mean()
            return majority_vote.item(), activation_value.item()

        return 1, 1.0

