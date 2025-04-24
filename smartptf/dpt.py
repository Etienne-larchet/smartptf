
import matplotlib.pyplot as plt
import polars as pl
from scipy.signal import welch


class DPT:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.log_returns = self._calculate_log_returns()

    def _calculate_log_returns(self) -> pl.DataFrame:
        close_prices = [col for col in self.df.columns if col.endswith('Close')]
        log_returns = self.df.with_columns( 
            [ (pl.col(col).log() - pl.col(col).log().shift(1)) for col in close_prices])
        return log_returns.drop_nulls()
    
    def apply_welch_method(self, nperseg: int = 48, noverlap: int = 24):
        f, Pxx = welch(self.log_returns, fs=1, nperseg=nperseg, noverlap=noverlap, scaling='density')

        plt.semilogy(f, Pxx)
        plt.grid(True)
        plt.show()
