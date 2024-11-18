import pandas as pd
from ta.utils import IndicatorMixin
import numpy as np


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