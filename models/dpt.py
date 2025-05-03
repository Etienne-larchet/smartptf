from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.signal import coherence, csd, welch


@dataclass
class DPT:
    data: pl.DataFrame
    index_ticker: str
    R: np.ndarray = field(default=None, init=False)
    csd: np.ndarray = field(default=None, init=False)
    coherence: np.ndarray = field(default=None, init=False)
    phase_shift: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        if self.data is None:
            raise ValueError("Data must be provided")
        self.data = self.data.with_columns(
            (pl.col(col).log() - pl.col(col).log().shift(1)).alias(col.replace("_Close", "_logR"))
            for col in self.data.columns
            if col.endswith("Close")
        )
        self._calculate_signals()

    def _calculate_signals(self, T: int = 48, dt: int = 1):
        fs = 1 / dt
        spectral_params = dict(fs=fs, nperseg=T, noverlap=T // 2, window="boxcar")

        # Calculate Fourier Transform
        log_returns = self.get("logR", include_index=True, include_date=False)
        _, self.R = welch(log_returns.transpose(), **spectral_params)
        self.R = self.R[:, 1:]  # Remove the first element (frequency 0)
        self.tickers = [col.removesuffix("_logR") for col in log_returns.columns]

        # Calculate phase-shift
        index_logR = log_returns[self.index_ticker]
        assets_logR = log_returns.select(pl.all().exclude(self.index_ticker)).transpose()
        ## Cross-Spectral Density
        _, self.csd = csd(assets_logR, index_logR, **spectral_params)
        self.csd = self.csd[:, 1:]  # Remove the first element (frequency 0)
        ## Coherence
        _, self.coherence = coherence(assets_logR, index_logR, **spectral_params)
        self.coherence = self.coherence[:, 1:]  # Remove the first element (frequency 0)
        ## Phase shift
        self.phase_shift = np.angle(self.csd)  # Phase from cross-spectrum

    def portfolio_variance(self, weights, phase_shift: float):
        _, K = self.R.shape
        variance = 0.0
        for k in range(K):
            beta_sum = np.sum(weights * self.R[k] * np.cos(phase_shift[k, :]))
            alpha_sum = np.sum(weights * self.R[k] * np.sin(phase_shift[k, :]))
            variance += beta_sum**2 + alpha_sum**2

        return 1 / 2 * variance

    def get(
        self, cat: Literal["Close", "logR"], include_index: bool = False, include_date: bool = True
    ) -> pl.DataFrame:
        query = [pl.col("Date")] if include_date else []
        query += [pl.col(f"{self.index_ticker}_{cat}")] if include_index else []
        query += [cs.ends_with(f"_{cat}") - cs.starts_with(self.index_ticker)]
        result = self.data.select(query).drop_nulls()
        return result.rename({col: col.replace(f"_{cat}", "") for col in result.columns})
