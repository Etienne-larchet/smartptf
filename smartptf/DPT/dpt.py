
import polars as pl


class Dpt:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.log_returns = self._calculate_log_returns()

    def _calculate_log_returns(self) -> pl.DataFrame:
        close_prices = [col for col in self.df.columns if col.endswith('Close')]
        log_returns = self.df.with_columns( 
            [ (pl.col(col).log() - pl.col(col).log().shift(1)) for col in close_prices])
        return log_returns.drop_nulls()
    
    def apply_welch_method(self, fs: float = 1.0, nperseg: int = 256):
        raise NotImplementedError("Welch method is not implemented yet.")