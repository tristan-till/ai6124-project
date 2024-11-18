import numpy as np
from ta.utils import IndicatorMixin

from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
import pandas as pd


class VRSIIndicator(IndicatorMixin):
    """Volume Relative Strength Index (VRSI)

    Compares the volume variations of recent gains and losses over a specified time
    period to measure speed and change of volume movements of a security. It is
    primarily used to attempt to identify the volume of overbought or oversold conditions in
    the trading of an asset.

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        close: pd.Series,
        volume: pd.Series,
        window: int = 14,
        fillna: bool = False,
    ):
        self._close = close
        self._volume = volume
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        diff = self._close.diff(1)
        up_direction = self._volume.where(diff > 0, 0.0)
        down_direction = self._volume.where(diff < 0, 0.0)
        min_periods = 0 if self._fillna else self._window
        emaup = up_direction.ewm(
            alpha=1 / self._window, min_periods=min_periods, adjust=False
        ).mean()
        emadn = down_direction.ewm(
            alpha=1 / self._window, min_periods=min_periods, adjust=False
        ).mean()
        relative_strength = emaup / emadn
        self._vrsi = pd.Series(
            np.where(emadn == 0, 100, 100 - (100 / (1 + relative_strength))),
            index=self._close.index,
        )

    def vrsi(self) -> pd.Series:
        """Volume Relative Strength Index (VRSI)

        Returns:
            pandas.Series: New feature generated.
        """
        vrsi_series = self._check_fillna(self._vrsi, value=50)
        return pd.Series(vrsi_series, name="vrsi")


def get_rsi(close, window=14):
    rsi = RSIIndicator(close, window=window)
    rsi = rsi.rsi()
    return rsi


def get_vrsi(close, volume, window=14):
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
    macd.name = "macd"
    return macd


def get_macd_diff(close):
    macd = MACD(close=close)
    macd_diff = macd.macd_diff()
    macd_diff.name = "macd_d"
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
    vmacd.name = "vmacd"
    return vmacd


def get_vmacd_signal(volume):
    _, dea, _ = calculate_vmacd(volume)
    dea.name = "vmacd_s"
    return dea


def calculate_bollinger_bands(close):
    bb = BollingerBands(close=close)
    return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()


def get_bb_high(close):
    bb_high, _, _ = calculate_bollinger_bands(close)
    bb_high.name = "bb_h"
    return bb_high


def get_bb_mid(close):
    _, bb_mid, _ = calculate_bollinger_bands(close)
    bb_mid.name = "bb_m"
    return bb_mid


def get_bb_low(close):
    _, _, bb_low = calculate_bollinger_bands(close)
    bb_low.name = "bb_l"
    return bb_low
