import numpy as np
import torch

from utils.fis import GenFIS

import utils.params as params
import utils.enums as enums
import utils.objectives as objectives

class PortfolioManager:
    def __init__(self, device, num_inputs, cash=params.INITIAL_CASH, stocks=params.INITIAL_STOCKS, transaction_fee=params.TRANSACTION_FEE, rule_operator=lambda x: sum(x) / len(x), mode=enums.Mode.TRAIN):
        self.device = device
        self.transaction_fee = transaction_fee

        self.num_default_inputs = params.NUM_SELF_INPUTS
        self.significance_threshold = params.SIGNIFICANCE_THRESHOLD
        
        self.fis = GenFIS(device=device, num_inputs=num_inputs + self.num_default_inputs, rule_operator=rule_operator)
        
        self.cash = torch.tensor(cash, device=device, dtype=torch.float32)
        self.num_stocks = torch.tensor(stocks, device=device, dtype=torch.float32)
        
        # Track actions and history as tensors
        self.num_actions = torch.tensor(0, device=device, dtype=torch.int32)
        self.history = torch.empty(0, device=device, dtype=torch.float32)
        self.buys = torch.empty(0, device=device, dtype=torch.int32)
        self.sells = torch.empty(0, device=device, dtype=torch.int32)
        self.mode = mode
        
    def forward(self, inputs, price, act=True):
        inputs = torch.cat((inputs, self.liquidity(price).unsqueeze(0))).to(self.device)
        actions = self.fis.forward(inputs)
        if act:
            self.act(actions, price)
        return actions
    
    def act(self, actions, price):
        sell = actions[0]
        hold = actions[1]
        buy = actions[2]

        if buy > hold and buy > sell:
            amount = torch.floor(self.cash * buy / price).to(torch.int32)
            self.buy(amount, price)
        elif sell > hold and sell > buy:
            amount = torch.floor(sell * self.num_stocks).to(torch.int32)
            self.sell(amount, price)

        value = self.get_value(price)
        self.history = torch.cat((self.history, value.unsqueeze(0)))
        self.num_actions += 1
            
    def buy(self, amount, price):
        self.buys = torch.cat((self.buys, self.num_actions.unsqueeze(0)))
        if (self.cash < amount * price or amount < 1) and self.mode == enums.Mode.TRAIN:
            self.cash -= 1
            return
        
        self.cash -= amount * price * (1 + self.transaction_fee)
        self.num_stocks += amount
        
    def sell(self, amount, price):
        self.sells = torch.cat((self.sells, self.num_actions.unsqueeze(0)))
        if (self.num_stocks < amount or amount < 1) and self.mode == enums.Mode.TRAIN:
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
    
    def eval(self):
        gain = objectives.cumulative_return(self.history)
        sharpe = objectives.sharpe_ratio(self.history)
        max_dd = objectives.maximum_drawdown(self.history, train=False)
        print(f"Buy and Hold: Gain={gain:.4f}, Sharpe={sharpe:.4f}, Max Drawdown={max_dd:.4f}")
    
    def get_history(self):
        return self.history, self.buys, self.sells
    
    def reset(self):
        self.cash = torch.tensor(params.INITIAL_CASH, device=self.device, dtype=torch.float32)
        self.num_stocks = torch.tensor(params.INITIAL_STOCKS, device=self.device, dtype=torch.float32)
        self.num_actions.zero_()
        self.history = torch.empty(0, device=self.device, dtype=torch.float32)
        self.buys = torch.empty(0, device=self.device, dtype=torch.int32)
        self.sells = torch.empty(0, device=self.device, dtype=torch.int32)