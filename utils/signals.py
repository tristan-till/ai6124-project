import pandas as pd
import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership

from mfs import *


# Base class for trading signals
class Signal:
    def generate_signal(self, prices):
        """Generate trading signals based on price data."""
        raise NotImplementedError("This method should be overridden by subclasses.")


# EMA Signal class
class EMASignal(Signal):
    def __init__(self, short_window=40, long_window=100):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, prices):
        signals = self._initialize_signals(prices)
        signals["short_ema"] = (
            prices["close"].ewm(span=self.short_window, adjust=False).mean()
        )
        signals["long_ema"] = (
            prices["close"].ewm(span=self.long_window, adjust=False).mean()
        )
        signals["signal"] = np.where(signals["short_ema"] > signals["long_ema"], 1, 0)
        return signals

    def _initialize_signals(self, prices):
        return pd.DataFrame(index=prices.index, data={"price": prices["close"]})


# RSI Signal class
class RSISignal(Signal):
    def __init__(self, period=14):
        self.period = period

    def generate_signal(self, prices):
        delta = prices["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        signals = self._initialize_signals(prices)
        signals["rsi"] = rsi
        signals["signal"] = np.where(rsi < 30, 1, np.where(rsi > 70, 0, np.nan))
        signals["signal"].ffill(inplace=True)  # Forward fill to avoid NaNs
        return signals

    def _initialize_signals(self, prices):
        return pd.DataFrame(index=prices.index, data={"price": prices["close"]})


# MACD Signal class
class VMACDSignal(Signal):
    def __init__(self, short_window=12, long_window=26, signal_window=9):
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

    def generate_signal(self, prices):
        short_ema = prices["close"].ewm(span=self.short_window, adjust=False).mean()
        long_ema = prices["close"].ewm(span=self.long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=self.signal_window, adjust=False).mean()

        signals = self._initialize_signals(prices)
        signals["macd"] = macd
        signals["signal_line"] = signal_line
        signals["signal"] = np.where(
            macd > signal_line, 1, np.where(macd < signal_line, 0, np.nan)
        )
        signals["signal"].ffill(inplace=True)  # Forward fill to avoid NaNs
        return signals

    def _initialize_signals(self, prices):
        return pd.DataFrame(index=prices.index, data={"price": prices["close"]})


# Fuzzy Signal class inheriting from Signal
class FuzzySignal(Signal):
    def __init__(self, feature_configs, X, y, transaction_fee=0.01, sensitivity=0.5):
        self.features = {}
        self.input_ranges = {}

        for feature_name, config in feature_configs.items():
            num_memberships = config.get("num_memberships", 3)
            membership_type = config.get("membership_type", "triangular")
            min_val = X[feature_name].min()
            max_val = X[feature_name].max()
            self.input_ranges[feature_name] = (min_val, max_val)
            self.features[feature_name] = ctrl.Antecedent(
                np.arange(min_val, max_val + 1e-5, (max_val - min_val) / 100),
                feature_name,
            )
            self._define_membership_functions(
                feature_name, num_memberships, membership_type
            )

        self.action = ctrl.Consequent(np.arange(-1, 1), "action")
        self.transaction_fee = transaction_fee
        self.sensitivity = sensitivity
        self._define_action_membership_functions(y)

    def _define_membership_functions(
        self, feature_name, num_memberships, membership_type
    ):
        """Define membership functions for a given feature."""

        if membership_type == "triangular":
            for i in range(num_memberships):
                center = (i + 1) * (
                    self.input_ranges[feature_name][1]
                    - self.input_ranges[feature_name][0]
                ) / (num_memberships + 1) + self.input_ranges[feature_name][0]
                width = (
                    self.input_ranges[feature_name][1]
                    - self.input_ranges[feature_name][0]
                ) / (num_memberships * 2)
                mf = TriangularMembershipFunction(
                    max(self.input_ranges[feature_name][0], center - width),
                    center,
                    min(self.input_ranges[feature_name][1], center + width),
                )
                self.features[feature_name][f"membership_{i}"] = mf.compute_membership

        elif membership_type == "trapezoidal":
            for i in range(num_memberships):
                left = (i) * (
                    self.input_ranges[feature_name][1]
                    - self.input_ranges[feature_name][0]
                ) / (num_memberships + 1) + self.input_ranges[feature_name][0]
                right = (i + 1) * (
                    self.input_ranges[feature_name][1]
                    - self.input_ranges[feature_name][0]
                ) / (num_memberships + 1) + self.input_ranges[feature_name][0]
                mf = TrapezoidalMembershipFunction(
                    left
                    - 0.05
                    * (
                        self.input_ranges[feature_name][1]
                        - self.input_ranges[feature_name][0]
                    ),
                    left,
                    right,
                    right
                    + 0.05
                    * (
                        self.input_ranges[feature_name][1]
                        - self.input_ranges[feature_name][0]
                    ),
                )
                self.features[feature_name][f"membership_{i}"] = mf.compute_membership

        elif membership_type == "gaussian":
            for i in range(num_memberships):
                mean = (i + 1) * (
                    self.input_ranges[feature_name][1]
                    - self.input_ranges[feature_name][0]
                ) / (num_memberships + 1) + self.input_ranges[feature_name][0]
                sigma = (
                    0.1
                    * (
                        self.input_ranges[feature_name][1]
                        - self.input_ranges[feature_name][0]
                    )
                    / num_memberships
                )
                mf = GaussianMembershipFunction(mean, sigma)
                self.features[feature_name][f"membership_{i}"] = mf.compute_membership

        else:
            raise ValueError("Unsupported membership function type.")

    def _define_rules(self):
        n_features = len(self.features)
        self.Lut_m = np.empty(n_features, np.int8)
        self.Lut_d = np.empty(n_features, np.int8)

        # Calculate lookup tables for rule access
        self.N_rules = 1
        for i, feature in enumerate(self.features):
            self.Lut_m[i] = len(feature)  # n_mf
            self.Lut_d[i] = 1
            self.N_rules *= self.Lut_m[0]
            for j in range(i):
                self.Lut_d[j] *= self.Lut_m[j]

    def _get_label(self, rule_no, input_no):
        return int((int(rule_no) / (self.Lut_d[input_no])) % self.Lut_m[input_no])

    def _define_action_membership_functions(self, y):
        min_y = y.min()
        max_y = y.max()

        hold_range_adjustment = max_y * self.transaction_fee * (1 + self.sensitivity)

        sell_threshold = min_y + hold_range_adjustment
        buy_threshold = max_y - hold_range_adjustment

        sell_threshold = max(sell_threshold, min_y)
        buy_threshold = min(buy_threshold, max_y)

        self.action["sell"] = membership.trimf(
            self.action.universe, [-np.inf, min_y, sell_threshold]
        )
        self.action["hold"] = membership.trimf(
            self.action.universe, [sell_threshold, 0, buy_threshold]
        )
        self.action["buy"] = membership.trimf(
            self.action.universe, [buy_threshold, max_y, np.inf]
        )

    def _learn_rules(self, X, y, epochs=100):
        # All datapoints
        for idx in range(len(X.iloc[0, :])):
            inputs = np.zeros((self.N_rules))
            outputs = np.zeros((3))
            for rule_no in range(self.N_rules):
                min_t = 1.0
                for input_no, feature_name in enumerate(self.features):
                    label = self._get_label(rule_no, input_no)
                    t = self.features[feature_name][f"membership_{label}"](
                        X[feature_name][idx]
                    )
                    min_t = min(min_t, t)
                    inputs[input_no] = min_t
                for output_no in range(3):
                    t = evalmf(self.Out_mf[0][output_no], y)
                    self.pweights[rule_no][output_no] += min_t * t

    def compute_signal(self, inputs):
        if not hasattr(self.action.universe, "value"):
            raise ValueError("Control system not built.")

        for feature_name in inputs.keys():
            if feature_name in self.features:
                value = inputs[feature_name]
                min_val, max_val = self.input_ranges[feature_name]
                normalized_value = (value - min_val) / (max_val - min_val)
                # Set input to simulation logic here

                # Compute output logic here

                return {
                    "action": None,
                    "action_type": "Hold",  # Placeholder for action type logic
                }


# Portfolio Manager class
class PortfolioManager:
    def __init__(self, initial_capital=100000.0, transaction_fee=0.01):
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.current_cash = initial_capital
        self.positions = pd.DataFrame(columns=["stock"])

    def set_signals(self, signal: Signal, prices):
        """Set the trading signals using a provided signal class."""
        self.signals = signal.generate_signal(prices)

    def buy(self, price: float, quantity: int):
        """Execute a buy order."""
        total_cost = price * quantity * (1 + self.transaction_fee)

        if total_cost <= self.current_cash:
            self.current_cash -= total_cost

            if "stock" not in self.positions.columns:
                self.positions["stock"] = 0

            self.positions["stock"] += quantity
            print(f"Bought {quantity} shares at {price} each.")

    def sell(self, price: float, quantity: int):
        """Execute a sell order."""

        if quantity <= self.positions.get("stock", [0]).sum():
            total_revenue = price * quantity * (1 - self.transaction_fee)
            self.current_cash += total_revenue

            if "stock" in self.positions.columns:
                self.positions["stock"] -= quantity

            print(f"Sold {quantity} shares at {price} each.")

    def backtest(self):
        """Backtest the portfolio performance based on signals."""

        for date in self.signals.index:
            if date in self.signals.index:
                signal_value = self.signals.loc[date]["signal"]
                price = self.signals.loc[date]["price"]

                if signal_value == 1:  # Buy signal
                    # Example: Buy a fixed number of shares for simplicity
                    self.buy(price, 10)
                elif (
                    signal_value == 0 and self.positions.get("stock", [0]).sum() > 0
                ):  # Sell signal
                    # Example: Sell all shares for simplicity
                    self.sell(price, int(self.positions.get("stock", [0]).sum()))


# Example usage of the PortfolioManager with EMASignal and FuzzySignal
if __name__ == "__main__":
    # Sample DataFrame with historical price data should be defined here.

    portfolio_manager_ema = PortfolioManager()
    ema_signal_instance = EMASignal(short_window=40)

    # Assuming 'prices_df' is defined previously with historical price data.
    portfolio_manager_ema.set_signals(ema_signal_instance, prices_df)

    # Backtest the strategy based on generated EMA signals.
    portfolio_manager_ema.backtest()

    # Example usage of FuzzySignal can be added here similarly.
