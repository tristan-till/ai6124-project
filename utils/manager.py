import numpy as np
import torch

from utils.fis import GenFIS

import utils.params as params

class PortfolioManager:
    def __init__(self, device, num_inputs, 
                 num_outputs=params.NUM_OUTPUTS, num_in_mf=params.NUM_IN_MF, num_out_mf=params.NUM_OUT_MF, 
                 cash=params.INITIAL_CASH, stocks=params.INITIAL_STOCKS, transaction_fee=params.TRANSACTION_FEE,
                 mutation_rate=params.MUTATION_RATE, rule_operator=lambda x: sum(x) / len(x)):
        self.device = device
        self.transaction_fee = transaction_fee

        self.num_default_inputs = params.NUM_SELF_INPUTS
        self.significance_threshold = params.SIGNIFICANCE_THRESHOLD
        
        self.fis = GenFIS(device, num_inputs + self.num_default_inputs, num_outputs, num_in_mf, num_out_mf, mutation_rate, rule_operator)
        
        self.cash = torch.tensor(cash, device=device, dtype=torch.float32)
        self.num_stocks = torch.tensor(stocks, device=device, dtype=torch.float32)
        
        # Track actions and history as tensors
        self.num_actions = torch.tensor(0, device=device, dtype=torch.int32)
        self.history = torch.empty(0, device=device, dtype=torch.float32)
        self.buys = torch.empty(0, device=device, dtype=torch.int32)
        self.sells = torch.empty(0, device=device, dtype=torch.int32)
        
    def forward(self, inputs, price):
        inputs = torch.cat((inputs, self.liquidity(price).unsqueeze(0))).to(self.device)
        
        actions = self.fis.forward(inputs)
        action = actions[0] - actions[1]
        is_significant = torch.abs(action) > self.significance_threshold

        if action > 0 and is_significant:
            amount = torch.floor(self.cash * torch.abs(action) / price).to(torch.int32)
            self.buy(amount, price)
        elif is_significant:
            amount = torch.floor(torch.abs(action) * self.num_stocks).to(torch.int32)
            self.sell(amount, price)

        value = self.get_value(price)
        self.history = torch.cat((self.history, value.unsqueeze(0)))
        self.num_actions += 1
            
    def buy(self, amount, price):
        self.buys = torch.cat((self.buys, self.num_actions.unsqueeze(0)))
        if self.cash < amount * price or amount < 1:
            self.cash -= 1
            return
        
        self.cash -= amount * price * (1 + self.transaction_fee)
        self.num_stocks += amount
        
    def sell(self, amount, price):
        self.sells = torch.cat((self.sells, self.num_actions.unsqueeze(0)))
        if self.num_stocks < amount or amount < 1:
            self.cash -= 1
            return
        
        self.cash += amount * price * (1 - self.transaction_fee)
        self.num_stocks -= amount

    def liquidity(self, price):
        value = self.get_value(price)
        return self.cash / value if value != 0 else torch.tensor(0.0, device=self.device)

    def get_value(self, price):
        return self.cash + self.num_stocks * price
        
    def print_portfolio(self, price):
        print("-" * 50)
        print("#### PORTFOLIO OVERVIEW ####")
        print("Total value: ", self.get_value(price).item())
        print(f"Number of stocks: {self.num_stocks.item()} worth {self.num_stocks * price}")
        print(f"Cash: {self.cash.item()}")
        print("-" * 50)
        return self.get_value(price)
    
    def get_history(self):
        return self.history, self.buys, self.sells
    
    def reset(self):
        self.cash = torch.tensor(params.INITIAL_CASH, device=self.device, dtype=torch.float32)
        self.num_stocks = torch.tensor(params.INITIAL_STOCKS, device=self.device, dtype=torch.float32)
        self.num_actions.zero_()
        self.history = torch.empty(0, device=self.device, dtype=torch.float32)
        self.buys = torch.empty(0, device=self.device, dtype=torch.int32)
        self.sells = torch.empty(0, device=self.device, dtype=torch.int32)