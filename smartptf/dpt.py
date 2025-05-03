
from dataclasses import dataclass, field
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.signal import coherence, csd, welch


@dataclass
class DPT:
    data: pl.DataFrame
    index_ticker: str
    R: pl.DataFrame = field(default=None, init=False)
    
    def __post_init__(self):
        if self.data is None:
            raise ValueError("Data must be provided")
        self.data = self.data.with_columns(
            (pl.col(col).log() - pl.col(col).log().shift(1)).alias(col.replace('_Close', '_logR'))
            for col in self.data.columns if col.endswith('Close')
        )
        self._process()
    
    def _process(self, T: int = 48, dt: int = 1, display_graph: bool = False):
        fs = 1/dt
        spectral_params = dict(fs=fs, nperseg=T, noverlap=T//2, window='boxcar')

        # Calculate Fourier Transform
        log_returns = self.get('logR', include_index=True, include_date=False)
        _, R_np = welch(log_returns.transpose(), **spectral_params)
        tickers = [col.removesuffix('_logR') for col in log_returns.columns]
        self.R = pl.DataFrame(R_np, schema=tickers)

        # Calculate phase-shift
        index_logR = log_returns[self.index_ticker]
        assets_logR = log_returns.select(pl.all().exclude(self.index_ticker)).transpose()
        ## Cross-Spectral Density                             
        _, Cxy = csd(assets_logR, index_logR, **spectral_params)
        ## Coherence
        _, Ccoh = coherence(assets_logR, index_logR, **spectral_params)
        ## Phase shift
        phase = np.angle(Cxy)  # Phase from cross-spectrum

        if display_graph:
            fig, axes = plt.subplots(4, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [1, 1.5]})
            x_full = np.tile(np.arange(R_np.shape[1]), R_np.shape[0])
            x_assets = np.tile(np.arange(R_np.shape[1]), R_np.shape[0]-1)

            # Power Spectral Density
            axes[0,0].bar(x_full, R_np.ravel(), alpha=0.01)
            axes[0,0].set_title('Welch Fourier Transform of returns')
            axes[0,0].set_xlabel('kth')
            axes[0,0].set_ylabel('Magnitude')
            axes[0,0].set_xlim(0)

            cax0 = axes[0,1].imshow(R_np, aspect='auto', cmap='viridis')
            axes[0,1].set_title("Heatmap")
            axes[0,1].set_xlabel("kth")
            axes[0,1].set_ylabel("Assets")
            fig.colorbar(cax0, label='SPD', ax=axes[0,1])

            #Cross-Spectral Density
            axes[1,0].bar(x_assets, Cxy.ravel(), alpha=0.01)
            axes[1,0].set_title('Cross-Spectral Density with Index')
            axes[1,0].set_xlabel('kth')
            axes[1,0].set_ylabel('Magnitude')
            axes[1,0].set_xlim(0)

            cax1 = axes[1,1].imshow(Cxy.real, aspect='auto', cmap='viridis')
            axes[1,1].set_xlabel("kth")
            axes[1,1].set_ylabel("Assets")
            fig.colorbar(cax1, label='SPD', ax=axes[1,1])

            # Coherence
            axes[2,0].bar(x_assets, Ccoh.ravel(), alpha=0.01)
            axes[2,0].set_title('Coherence with Index')
            axes[2,0].set_xlabel('kth')
            axes[2,0].set_ylabel('Magnitude')
            axes[2,0].set_xlim(0)

            cax2 = axes[2,1].imshow(Ccoh.real, aspect='auto', cmap='viridis')
            axes[2,1].set_xlabel("kth")
            axes[2,1].set_ylabel("Assets")
            fig.colorbar(cax2, label='SPD', ax=axes[2,1])

            # Phase shift
            axes[3,0].bar(x_assets, phase.ravel(), alpha=0.01)
            axes[3,0].set_title('Phase Shift with Index')
            axes[3,0].set_xlabel('kth')
            axes[3,0].set_ylabel('Magnitude')
            axes[3,0].set_xlim(0)

            cax2 = axes[3,1].imshow(phase.real, aspect='auto', cmap='viridis')
            axes[3,1].set_xlabel("kth")
            axes[3,1].set_ylabel("Assets")
            fig.colorbar(cax2, label='SPD', ax=axes[3,1])

            plt.tight_layout(pad=0.5, h_pad=2.0)
            plt.show()

    def portfolio_variance(self, weights, phase_shift: float):

        _, K = self.R.shape
        variance = 0.0
        for k in range(K):
            beta_sum = np.sum(weights * self.R[k] * np.cos(phase_shift[k, :]))
            alpha_sum = np.sum(weights * self.R[k] * np.sin(phase_shift[k, :]))
            variance += beta_sum**2 + alpha_sum**2

        return 1/2 * variance



    def get(self, cat: Literal['Close', 'logR'], include_index: bool = False, include_date: bool = True) -> pl.DataFrame:
        query = [pl.col('Date')] if include_date else []
        query += [pl.col(f'{self.index_ticker}_{cat}')] if include_index else []
        query += [cs.ends_with(f'_{cat}') - cs.starts_with(self.index_ticker)]
        result = self.data.select(query).drop_nulls()
        return result.rename({col: col.replace(f'_{cat}','') for col in result.columns})
