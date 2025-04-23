import logging
from datetime import date
from pathlib import Path
from typing import override

import polars as pl
import yfinance as yf
from pydantic import BaseModel, Field, PrivateAttr, validate_call

from smartptf.config import DATA_DIR

from .utils import Period, relativedelta_str

logger = logging.getLogger(__name__)


class Indice(BaseModel):
    model_config = {'extra': 'allow'}

    name: str
    csv_compo_path: str | Path
    date_end: date  | None = None
    date_start: date | None = None # Either specify a starting date either specify a period
    period: Period | None  = None
    compo: list[str] = Field(default=None, init=False)
    _data: pl.DataFrame = PrivateAttr(default=None)
    csv_data_path: str = Field(default=None, init=False)

    @override
    def model_post_init(self, _) -> None:
        if self.date_end is None:
            self.date_end = date.today()
        if self.period is None and self.date_start is None:
            raise ValueError("Either date_start or period must be specified")
        if self.period is not None and self.date_start is not None:
            raise ValueError("Either date_start or period must be specified, not both")
        if self.period is not None:
            self.date_start = self.date_end - relativedelta_str(self.period)

        self.compo = self.get_composition(self.date_end)

    @validate_call(validate_return=True)
    def get_composition(self, date_ref: date) -> list[str]:
        df = pl.read_csv(self.csv_compo_path, schema_overrides={"date": date})
        tickers_str = (
            df.filter(pl.col("date") <= date_ref)
            .sort("date")
            .tail(1)
            .select("tickers")
            .item()
        )
        return tickers_str.split(",")

    @validate_call
    def load_from_yahoo(self, threshold_missing_val: float = 0.03) -> None:
        df = yf.download(
            self.compo,
            start=self.date_start,
            end=self.date_end,
            interval="1mo",
            auto_adjust=True,
            group_by="ticker",
            keepna=True # Keep control on missing values
        )
        tickers = df.columns.levels[0]
        logger.info(f"Data loaded for {len(tickers)} tickers")
        drop_list = { ticker for ticker in tickers if df[ticker, "Close"].isna().mean() > threshold_missing_val}
        if drop_list:
            logger.info(f"{len(drop_list)} tickers dropped due to missing data > {threshold_missing_val:.2%}")
            logger.debug(f"Tickers dropped: {drop_list}")
        else:
            logger.debug("No tickers dropped")
        df2 = df.drop(columns=drop_list, level=0)
        col_names = [f"{ticker}_{col}" for ticker, col in df2.columns]
        df2.columns = col_names
        df3 = df2.reset_index() # Transform index Date into a column, to be extracted by polars
        self._data = pl.from_pandas(df3)

        logger.info("Data transformed to Polars DataFrame")

    def load_from_csv(self, directory: str | Path | None = None) -> None:
        try:
            if directory is None:
                directory = DATA_DIR
            elif isinstance(directory, str):
                directory = Path(directory)

            self.csv_data_path = directory / f"{self.name}_ohlcv_{self.date_start}_to_{self.date_end}.csv"
            self._data = pl.read_csv(self.csv_data_path)
            logger.info(f'Data loaded from "{self.csv_data_path}"')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {self.csv_data_path}") from e
        except OSError as e:
            raise OSError(f"IO error while loading data from {self.csv_data_path}") from e
    
    def load_from_eodhd(self) -> None:
        raise NotImplementedError("EODHD API loading not implemented yet")
    
    def to_csv(self, directory: str | Path | None = None) -> None:
        if self._data is None:
            raise ValueError("Data is not loaded yet")
        if directory is None:
            directory = DATA_DIR
        elif isinstance(directory, str):
            directory = Path(directory)

        self.csv_data_path = directory / f"{self.name}_ohlcv_{self.date_start}_to_{self.date_end}.csv"
        try:
            self._data.write_csv(self.csv_data_path)
            logger.info(f'Data saved to "{self.csv_data_path}"')
        except Exception as e:
            raise RuntimeError("An error occurred while saving data to CSV") from e

    @property
    def open(self) -> pl.DataFrame:
        return self._data.select((pl.col('Date'), pl.col('^.*_Open$')))
    @property
    def close(self) -> pl.DataFrame:
        return self._data.select((pl.col('Date'), pl.col('^.*_Close$')))
    @property
    def low(self) -> pl.DataFrame:
        return self._data.select((pl.col('Date'), pl.col('^.*_Low$')))
    @property
    def high(self) -> pl.DataFrame:
        return self._data.select((pl.col('Date'), pl.col('^.*_High$')))
    @property
    def volume(self) -> pl.DataFrame:
        return self._data.select((pl.col('Date'), pl.col('^.*_Volume$')))
