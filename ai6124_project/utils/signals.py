import pandas as pd
import numpy as np

from ta.trend import EMAIndicator
import ai6124_project.utils.indicators as indicators

class Signal:
    def generate_signal(self, prices):
        """Generate trading signals based on price data."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
def get_ema_signal(close):
    ema_short = indicators.get_ema(close, 12)
    ema_long =  indicators.get_ema(close, 26)
    return ema_short - ema_long

def get_rsi_signal(close):
    overbought_threshold = 0.7
    oversold_threshold = 0.3
    short_rsi = indicators.get_normalized_rsi(close, 7, overbought_threshold, oversold_threshold)
    med_rsi = indicators.get_normalized_rsi(close, 30, overbought_threshold, oversold_threshold)
    long_rsi = indicators.get_normalized_rsi(close, 90, overbought_threshold, oversold_threshold)
    rsi_signal = (short_rsi + med_rsi + long_rsi) / 3
    return rsi_signal

def get_macd_signal(close):
    # Compute MACD and Signal Line
    macd_indicator = indicators.get_macd_incidator(close)
    macd_line = macd_indicator.macd()
    signal_line = macd_indicator.macd_signal()
    macd_signal = 1 / (1 + np.exp(-10 * (macd_line - signal_line)))
    return macd_signal

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