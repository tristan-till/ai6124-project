import numpy as np

from utils.fis import GenFIS

class PortfolioManager:
    def __init__(self, cash, transaction_fee,
                 num_inputs, num_outputs, num_in_mf, num_out_mf, 
                 mutation_rate=0.1, rule_operator=lambda x: sum(x) / len(x)):
        self.initial_cash = cash
        self.transaction_fee = transaction_fee

        self.num_default_inputs = 1
        
        self.fis = GenFIS(num_inputs + self.num_default_inputs, num_outputs, num_in_mf, num_out_mf, mutation_rate, rule_operator)
        self.fis.randomize()
        
        self.cash = cash
        self.num_stocks = 0

        self.num_actions = 0
        self.history = []
        self.buys = []
        self.sells = []
        
    def forward(self, inputs, price):
        inputs = np.append(inputs, self.liquidity(price))
        actions = self.fis.forward(inputs)
        action = actions[0] - actions[1]
        is_significant = abs(action) > 0.2
        if action > 0 and is_significant:
            amount = int(self.cash * abs(action) / price)
            self.buy(amount, price)
        elif is_significant:
            amount = int(abs(action) * self.num_stocks)
            self.sell(amount, price)
        value = self.get_value(price)
        self.history.append(value)
        self.num_actions += 1
            
    def buy(self, amount, price):
        if self.cash < amount*price or amount < 1:
            self.cash -= 5
            return
        self.cash -= amount * price * (1+self.transaction_fee)
        self.num_stocks += amount
        self.buys.append(self.num_actions)
        
    def sell(self, amount, price):
        if self.num_stocks < amount*price or amount < 1:
            self.cash -= 5
            return
        self.cash += amount * price * (1-self.transaction_fee)
        self.num_stocks -= amount
        self.sells.append(self.num_actions)

    def liquidity(self, price):
        value = self.get_value(price)
        if value == 0:
            return 0
        return self.cash / self.get_value(price)
        
    def get_value(self, price):
        return self.cash + self.num_stocks * price
        
    def print_portfolio(self, price):
        print("-" * 50)
        print("#### PORTFOLIO OVERVIEW ####")
        print("Total value: ", self.get_value(price))
        print(f"Number of stocks: {self.num_stocks} worth {self.num_stocks*price}")
        print(f"Cash: {self.cash}")
        print("-" * 50)
        return self.cash + self.num_stocks * price
    
    def get_history(self):
        return self.history, self.buys, self.sells
    
    def reset(self):
        self.cash = self.initial_cash
        self.num_stocks = 0
        self.num_actions = 0
        self.history = []
        self.buys = []
        self.sells = []