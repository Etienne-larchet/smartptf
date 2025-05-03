import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from io import StringIO
from pathlib import Path
from time import sleep
from typing import Literal, override

import polars as pl
import requests
import yfinance as yf
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call
from tqdm import tqdm

from utils.utils import Period, force_list, relativedelta_str

logger = logging.getLogger(__name__)


@dataclass
class Eodhd:
    api_key: str = field(repr=False)
    base_url: str = "https://eodhd.com/api"

    @force_list("tickers")
    def get_historical(
        self,
        tickers: str | list[str],
        from_date: date | str,
        to_date: date | str,
        interval: Literal["d", "w", "m"] = "m",
        order: Literal["a", "d"] = "a",
        fmt: Literal["csv", "json"] = "csv",
        display_progress: bool = True,
    ) -> pl.DataFrame:
        params = {
            "api_token": self.api_key,
            "period": interval,  # Change of lexic for convention ('period' in eodhd being equivalent to 'interval' in yfinance)
            "from": from_date,
            "to": to_date,
            "order": order,
            "fmt": fmt,
        }
        full_lf = pl.DataFrame(schema={"Date": pl.Date}).lazy()
        missing_tickers = []
        logger.info("Fetching historical prices from EODHD API...")
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._get_historical_one_thread, ticker=ticker, params=params): ticker
                for ticker in tickers
            }  # noqa: E501
            with tqdm(
                total=len(tickers), desc="Fetching historical prices", unit="ticker", disable=not display_progress
            ) as pbar:
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        df = future.result()
                        lf = df.lazy()
                        full_lf = full_lf.join(lf, on="Date", how="full", coalesce=True)
                    except Exception as e:
                        logger.debug(f"Fetching data with ticker: {ticker} generated an exception: {e}")
                        missing_tickers.append(ticker)
                    finally:
                        pbar.update(1)
        full_lf = full_lf.sort("Date", descending=False)
        if interval == "m":
            full_lf = full_lf.with_columns(pl.col("Date").dt.month_end().alias("Date"))
            colnames = full_lf.collect_schema().names()
            expressions = [pl.col(c).filter(pl.col(c).is_not_null()).first().alias(c) for c in colnames[1:]]
            full_lf = (
                full_lf.group_by("Date").agg(expressions).sort("Date", descending=False)
            )  # Re sorting after group_by to ensure order of date
        if missing_tickers:
            logger.warning(f"Missing tickers: {missing_tickers}")
        full_df = full_lf.collect()
        logger.info(f"{len(full_df)} tickers fetched from EODHD API")
        return full_df

    def _get_historical_one_thread(self, ticker: str, params: dict) -> pl.DataFrame:
        url = f"{self.base_url}/eod/{ticker}"
        res = requests.get(url, params=params)
        match res.status_code:
            case 429:
                logger.warning(f"Rate limit exceeded for ticker {ticker}. Retrying in 20 seconds...")
                sleep(20)
                return self._get_historical_one_thread(ticker, params)
            case 200:
                ...
            case _:
                raise requests.exceptions.HTTPError(
                    f"Error code {res.status_code} while fetching data from ticker {ticker}: {res.text}"
                )
        if params["fmt"] == "csv":
            df = pl.read_csv(StringIO(res.text), schema_overrides={"Date": date})

            adj_coef = df["Adjusted_close"] / df["Close"]
            df2 = df.select(
                [
                    pl.col("Date"),
                    pl.col("Open") * adj_coef,
                    pl.col("High") * adj_coef,
                    pl.col("Low") * adj_coef,
                    pl.col("Adjusted_close").alias("Close"),
                    pl.col("Volume"),
                ]
            )

            df2.columns = [df2.columns[0]] + [f"{ticker}_{col}" for col in df2.columns[1:]]
            return df2
        else:
            raise ValueError("JSON format not implemented yet")


class Indice(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    csv_compo_path: str | Path
    date_end: date | None = None
    date_start: date | None = None  # Either specify a starting date either specify a period
    period: Period | None = None
    eodhd_key: str | None = None
    compo: list[str] = Field(default=None, init=False)
    csv_data_path: str = Field(default=None, init=False)
    _data: pl.DataFrame = PrivateAttr(default=None)
    _eodhd_client: Eodhd = PrivateAttr(default=None)

    @override
    def model_post_init(self, _) -> None:
        # Date End conversion
        if self.date_end is None or self.date_end > date.today():
            self.date_end = date.today()
        if (
            self.date_end + relativedelta(days=1)
        ).month == self.date_end.month:  # Making sure date_end is not already the last day of the month
            self.date_end = self.date_end + relativedelta(day=1, days=-1)  # Get the last day of the month

        # Date Start conversion
        if self.period is None and self.date_start is None:
            raise ValueError("Either date_start or period must be specified")
        if self.period is not None and self.date_start is not None:
            raise ValueError("Either date_start or period must be specified, not both")
        if self.period is not None:
            self.date_start = (
                self.date_end - relativedelta_str(self.period) - relativedelta(months=1)
            )  # Add one month to get exact period of returns

        # Init of EODHD client
        if self.eodhd_key is not None:
            self._eodhd_client = Eodhd(self.eodhd_key)

        # Getting Composition of the index
        self.compo = self.get_composition(self.date_end)

    @validate_call(validate_return=True)
    def get_composition(self, date_ref: date) -> list[str]:
        df = pl.read_csv(self.csv_compo_path, schema_overrides={"date": date})
        tickers_str = df.filter(pl.col("date") <= date_ref).sort("date").tail(1).select("tickers").item()
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
            keepna=True,  # Keep control on missing values
        )
        tickers = df.columns.levels[0]
        drop_list = {ticker for ticker in tickers if df[ticker, "Close"].isna().mean() > threshold_missing_val}
        if drop_list:
            logger.info(f"{len(drop_list)} tickers dropped due to missing data > {threshold_missing_val:.2%}")
            logger.debug(f"Tickers dropped: {drop_list}")
        else:
            logger.debug("No tickers dropped")
        df2 = df.drop(columns=drop_list, level=0)
        col_names = [f"{ticker}_{col}" for ticker, col in df2.columns]
        df2.columns = col_names
        df3 = df2.reset_index()  # Transform index Date into a column, to be extracted by polars
        self._data = pl.from_pandas(df3)

        logger.info("Data transformed to Polars DataFrame")

    def load_from_csv(self, directory: str | Path | None = None) -> None:
        try:
            if directory is None:
                directory = Path("data/")
            elif isinstance(directory, str):
                directory = Path(directory)
            self.csv_data_path = directory / f"{self.name}_ohlcv_{self.date_start}_to_{self.date_end}.csv"
            self._data = pl.read_csv(self.csv_data_path, schema_overrides={"Date": date})
            logger.info(f'Data loaded from "{self.csv_data_path}"')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {self.csv_data_path}") from e
        except OSError as e:
            raise OSError(f"IO error while loading data from {self.csv_data_path}") from e

    def load_from_eodhd(self, threshold_missing_val: float = 0.03, display_progress: bool = True) -> None:
        if self._eodhd_client is None:
            raise ValueError("EODHD API key was not provided")
        self._data = self._eodhd_client.get_historical(
            tickers=self.compo, from_date=self.date_start, to_date=self.date_end, display_progress=display_progress
        )
        dropping_tickers = [
            col.removesuffix("_Close")
            for col in self.close.columns
            if self.close[col].is_null().mean() > threshold_missing_val
        ]
        logger.warning(f"{len(dropping_tickers)} tickers dropped due to missing data > {threshold_missing_val:.2%}")
        logger.debug(f"Tickers dropped: {dropping_tickers}")
        self._data = self._data.select(col for col in self._data.columns if col.split("_")[0] not in dropping_tickers)

    def to_csv(self, directory: str | Path | None = None) -> None:
        if self._data is None:
            raise ValueError("Data is not loaded yet")
        if directory is None:
            directory = Path("data/")
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
        return self._data.select((pl.col("Date"), pl.col("^.*_Open$")))

    @property
    def close(self) -> pl.DataFrame:
        return self._data.select((pl.col("Date"), pl.col("^.*_Close$")))

    @property
    def low(self) -> pl.DataFrame:
        return self._data.select((pl.col("Date"), pl.col("^.*_Low$")))

    @property
    def high(self) -> pl.DataFrame:
        return self._data.select((pl.col("Date"), pl.col("^.*_High$")))

    @property
    def volume(self) -> pl.DataFrame:
        return self._data.select((pl.col("Date"), pl.col("^.*_Volume$")))

    @override
    def __hash__(self) -> int:
        return hash((self.name, self.date_start, self.date_end, self.period, self.eodhd_key))
