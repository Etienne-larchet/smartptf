from collections.abc import Generator
from dataclasses import dataclass
from datetime import date
from typing import Literal

import polars as pl
import polars.selectors as cs

TimesSeriesCol = Literal["Close", "Low", "High", "Close", "Volume", "logR"]


@dataclass(kw_only=True)
class TimesSeriesPolars:
    data: pl.DataFrame
    index_ticker: str | None = None

    def get(
        self, cat: TimesSeriesCol | None = None, include_index: bool = False, include_date: bool = True
    ) -> pl.DataFrame:
        query = cs.all()
        if not include_date:
            query -= cs.date()
        if self.index_ticker is not None and not include_index:
            query -= cs.starts_with(self.index_ticker)
        if cat:
            query -= ~cs.ends_with(f"_{cat}") & ~cs.date()

        result = self.data.select(query).drop_nulls()
        return result.rename({col: col.replace(f"_{cat}", "") for col in result.columns})

    def calculate_logR(self, enforce: bool = False) -> None:
        if any(col.endswith("_logR") for col in self.data.columns) and not enforce:
            return
        self.data = self.data.with_columns(
            (pl.col(col).log().diff()).alias(col.replace("_Close", "_logR"))
            for col in self.data.columns
            if col.endswith("_Close")
        )
        sorted_columns = ["Date"] + sorted(self.get(include_date=False, include_index=True).columns)
        self.data = self.data.select(sorted_columns)


def sliding_window(df: pl.DataFrame, first_date_limit: date, window_length: int = 193) -> Generator[pl.DataFrame]:
    df = df.sort("Date")

    total_rows = len(df)
    filtered = df.filter(pl.col("Date") <= first_date_limit)
    first_end_idx = len(filtered) - 1
    windows_nb = total_rows - first_end_idx
    if window_length > first_end_idx + 1:
        raise ValueError(
            f"Missing {window_length - (first_end_idx + 1)} timeseries with the specified parameters. "
            f"(window_length={window_length}, first_date_limit={first_date_limit}). "
            "Please provide a later date or a smaller window_length."
        )

    for i in range(windows_nb):
        idx = first_end_idx - window_length + 1 + i
        yield df.slice(idx, window_length)
