from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
import numpy as np

from classes.vrsi_indicator import VRSIIndicator

def get_rsi(close, window):
    rsi = RSIIndicator(close, window)
    rsi = rsi.rsi()
    return rsi

def get_vrsi(close, volume, window):
    vrsi = VRSIIndicator(close, volume, window=window)
    vrsi = vrsi.vrsi()
    return vrsi

def get_roc(close):
    roc = ROCIndicator(close=close)
    roc = roc.roc()
    return roc

def get_macd(close):
    macd = MACD(close=close)
    macd = macd.macd()
    macd.name = 'macd'
    return macd

def get_macd_diff(close):
    macd = MACD(close=close)
    macd_diff = macd.macd_diff()
    macd_diff.name = 'macd_d'
    return macd_diff

def calculate_vmacd(volume, short_period=12, long_period=26, signal_period=9):
    ema_short = EMAIndicator(close=volume, window=short_period).ema_indicator()
    ema_long = EMAIndicator(close=volume, window=long_period).ema_indicator()
    diff = ema_short - ema_long
    dea = EMAIndicator(close=diff, window=signal_period).ema_indicator()
    vmacd = diff - dea
    return vmacd, dea, diff

def get_vmacd(volume):
    vmacd, _, _ = calculate_vmacd(volume)
    vmacd.name = 'vmacd'
    return vmacd

def get_vmacd_signal(volume):
    _, dea, _ = calculate_vmacd(volume)
    dea.name = 'vmacd_s'
    return dea

def calculate_bollinger_bands(close):
    bb = BollingerBands(close=close)
    return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()

def get_bb_high(close):
    bb_high, _, _ = calculate_bollinger_bands(close)
    bb_high.name = 'bb_h'
    return bb_high

def get_bb_mid(close):
    _, bb_mid, _ = calculate_bollinger_bands(close)
    bb_mid.name = 'bb_m'
    return bb_mid

def get_bb_low(close):
    _, _, bb_low = calculate_bollinger_bands(close)
    bb_low.name = 'bb_l'
    return bb_low

def get_ema(series, window):
    return EMAIndicator(close=series, window=window).ema_indicator()

def get_macd_incidator(close):
    return MACD(close=close)

def get_normalized_rsi(close, window, overbought_threshold, oversold_threshold):
    rsi = get_rsi(close, window)
    return normalize_rsi(rsi, overbought_threshold, oversold_threshold)

def normalize_rsi(rsi, overbought_threshold, oversold_threshold):
        rsi = rsi / 100
        return np.clip((rsi - oversold_threshold) / (overbought_threshold - oversold_threshold), 0, 1)